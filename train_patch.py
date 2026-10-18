"""Train the flow32 generator with a weighted 16x16 patch discriminator.

J_G = 6 * weighted_adv + lambda * max-pooled MSE.
The discriminator is pretrained on frozen evolution forecasts, then the
adversarial phase runs for a fixed number of epochs. Every epoch is saved.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from sketch.data import make_loader
from sketch.ncwrap import SCALE, SketchFlowGen, build_flow, build_generator
from sketch.normalize import NormStats
from sketch.patch_disc import PatchDiscriminator
from sketch.patch_loss import maxpool_mse, sample_patches, weighted_patch_bce

STAGE = "patch"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def append_metrics(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _csi(pred: torch.Tensor, gt: torch.Tensor, thr: float) -> float:
    pb = pred >= thr
    tb = gt >= thr
    hits = torch.logical_and(pb, tb).sum()
    miss = torch.logical_and(tb, torch.logical_not(pb)).sum()
    fa = torch.logical_and(pb, torch.logical_not(tb)).sum()
    denom = hits + miss + fa
    if float(denom) == 0:
        return float("nan")
    return float(hits / denom)


@torch.no_grad()
def evaluate_gen(wrap: SketchFlowGen, loader, device: torch.device) -> dict:
    wrap.eval()
    mae_sum = 0.0
    buckets = {8.0: [], 51.2: [], 76.8: [], 102.4: []}
    n = 0
    for batch in loader:
        batch = move(batch, device)
        _fields, traj = wrap(batch)
        pred = traj["radar"].clamp(0.0, 1.0) * SCALE
        gt = batch["radar_gt"].clamp(0.0, 1.0) * SCALE
        mae_sum += float((pred - gt).abs().mean())
        for thr in buckets:
            c = _csi(pred, gt, thr)
            if math.isfinite(c):
                buckets[thr].append(c)
        n += 1
    wrap.train()
    wrap.freeze_flow()

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "radar_mae128": mae_sum / max(n, 1),
        "csi8": mean(buckets[8.0]),
        "csi20": mean(buckets[51.2]),
        "csi30": mean(buckets[76.8]),
        "csi40": mean(buckets[102.4]),
        "n_batches": n,
    }


def pack_ckpt(wrap: SketchFlowGen, disc, opt_g, opt_d, epoch: int, extra: dict) -> dict:
    return {
        "model": wrap.flow.state_dict(),
        "gen": wrap.gen.state_dict(),
        "disc": disc.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "epoch": epoch,
        "stage": STAGE,
        **extra,
    }


def patch_kwargs(tcfg: dict) -> dict:
    return {
        "n_frames": int(tcfg.get("patch_frames", 4)),
        "n_biased": int(tcfg.get("patch_biased", 4)),
        "n_uniform": int(tcfg.get("patch_uniform", 4)),
        "patch": int(tcfg.get("patch_size", 16)),
        "temperature": float(tcfg.get("patch_temperature", 32.0)),
    }


def pretrain_discriminator(wrap, disc, opt_d, loader, device, tcfg, metrics_path, grad_clip: float) -> None:
    """Fake samples are the frozen evolution field. Stop once D is clearly ahead, before saturation."""
    max_steps = int(tcfg.get("pretrain_max_steps", 0))
    stop_d = float(tcfg.get("pretrain_stop_d", 0.9))
    min_gap = float(tcfg.get("pretrain_min_gap", 1.0))
    min_steps = int(tcfg.get("pretrain_min_steps", 100))
    log_every = int(tcfg.get("log_every", 20))
    pk = patch_kwargs(tcfg)
    patch = pk.pop("patch")
    disc.train()
    wrap.gen.eval()
    recent_d: list[float] = []
    recent_gap: list[float] = []
    print(
        f"pretrain D on evolution vs GT, stop when D<{stop_d:g} and logit gap>{min_gap:g} "
        f"after {min_steps} steps",
        flush=True,
    )
    step = 0
    for batch in tqdm(loader, desc="pretrain D"):
        step += 1
        if max_steps and step > max_steps:
            break
        batch = move(batch, device)
        _hist, evo, _fields, _traj = wrap.evolve(batch)
        real = batch["radar_gt"].clamp(0.0, 1.0) * SCALE
        spec = sample_patches(real, patch=patch, **pk)
        loss_d, real_logit, fake_logit = weighted_patch_bce(
            disc, real, evo.detach(), spec, patch=patch, for_generator=False
        )
        if not torch.isfinite(loss_d):
            continue
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
        opt_d.step()
        d_f = float(loss_d.detach())
        gap = float(real_logit - fake_logit)
        recent_d.append(d_f)
        recent_gap.append(gap)
        if step % log_every == 0:
            window = recent_d[-log_every:]
            gap_w = recent_gap[-log_every:]
            md = sum(window) / len(window)
            mg = sum(gap_w) / len(gap_w)
            tqdm.write(
                f"[pretrain s{step}] D={d_f:.4f} meanD={md:.4f} real={float(real_logit):.3f} "
                f"fake={float(fake_logit):.3f} gap={mg:.3f}"
            )
            append_metrics(
                metrics_path,
                {
                    "split": "pretrain_step",
                    "step": step,
                    "stage": STAGE,
                    "gan_d": d_f,
                    "real_logit": float(real_logit),
                    "fake_logit": float(fake_logit),
                    "gap": gap,
                },
            )
            if step >= min_steps and md < stop_d and mg > min_gap:
                print(
                    f"pretrain stop at step {step}: meanD={md:.4f} gap={mg:.3f}",
                    flush=True,
                )
                return
    print(f"pretrain finished {step} steps without the stop rule", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_rp_flow32_gen_patch.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    use_precip = bool(dcfg.get("use_precip", False))
    ckpt_dir = Path(tcfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"

    src_norm = Path(tcfg["norm_stats"])
    dst_norm = ckpt_dir / "norm_stats.json"
    if not dst_norm.exists():
        shutil.copyfile(src_norm, dst_norm)
    norm = NormStats.load(dst_norm)

    train_loader = make_loader(
        dcfg["train_root"], dcfg["train_months"], norm,
        dcfg["t_in"], dcfg["t_out"], tuple(dcfg["image_hw"]),
        dcfg["batch_size"], dcfg["num_workers"], True,
        use_precip=use_precip,
    )
    val_loader = make_loader(
        dcfg["eval_root"], dcfg["val_months"], norm,
        dcfg["t_in"], dcfg["t_out"], tuple(dcfg["image_hw"]),
        dcfg["batch_size"], dcfg["num_workers"], False,
        use_precip=use_precip,
    )
    print(
        f"train windows={len(train_loader.dataset)} val windows={len(val_loader.dataset)} device={device}",
        flush=True,
    )

    flow = build_flow(cfg, device)
    flow_ckpt = Path(tcfg["flow_ckpt"])
    state = torch.load(flow_ckpt, map_location=device, weights_only=False)
    missing, unexpected = flow.load_state_dict(state["model"], strict=False)
    if missing or unexpected:
        print(f"flow load non-strict missing={list(missing)} unexpected={list(unexpected)}", flush=True)
    gen = build_generator(cfg).to(device)
    wrap = SketchFlowGen(flow, gen).to(device)
    n_flow = sum(p.numel() for p in wrap.flow.parameters())
    n_gen = sum(p.numel() for p in wrap.gen.parameters() if p.requires_grad)
    print(f"frozen flow params={n_flow} trainable gen params={n_gen} from {flow_ckpt}", flush=True)

    disc = PatchDiscriminator(base_c=int(tcfg.get("patch_disc_base_c", 32))).to(device)
    gen_lr = float(tcfg.get("gen_lr", 3e-5))
    opt_g = torch.optim.Adam(wrap.gen.parameters(), lr=gen_lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=gen_lr)
    beta = float(tcfg.get("beta_adv", 6.0))
    lambda_pool = float(tcfg.get("lambda_pool_mse", 0.02))
    ensemble_k = max(int(tcfg.get("ensemble_k", 4)), 1)
    pk = patch_kwargs(tcfg)
    patch = pk["patch"]
    print(
        f"patch D params={sum(p.numel() for p in disc.parameters())} lr={gen_lr:g} "
        f"beta={beta:g} lambda_pool_mse={lambda_pool:g} K={ensemble_k} "
        f"frames={pk['n_frames']} biased={pk['n_biased']} uniform={pk['n_uniform']} "
        f"patch={patch} temp={pk['temperature']:g}",
        flush=True,
    )

    grad_clip = float(tcfg.get("grad_clip", 1.0))
    pretrain_discriminator(wrap, disc, opt_d, train_loader, device, tcfg, metrics_path, grad_clip)

    epochs = int(tcfg.get("epochs", 12))
    log_every = int(tcfg.get("log_every", 20))
    best_csi40 = -1.0
    wrap.train()
    wrap.freeze_flow()
    sample_kw = {k: pk[k] for k in ("n_frames", "n_biased", "n_uniform", "temperature")}

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"epoch {epoch} [{STAGE}]")
        run_d = run_g = run_adv = run_pool = 0.0
        run_n = n_skip = 0
        for step, batch in enumerate(pbar, 1):
            batch = move(batch, device)
            hist, evo, _fields, _traj = wrap.evolve(batch)
            real = batch["radar_gt"].clamp(0.0, 1.0) * SCALE
            spec = sample_patches(real, patch=patch, **sample_kw)

            wrap.gen.eval()
            with torch.no_grad():
                fake_d = wrap.generate(hist, evo).clamp(0.0, SCALE)
            wrap.gen.train()
            disc.train()
            for p in disc.parameters():
                p.requires_grad = True
            loss_d, _real_logit, _fake_logit = weighted_patch_bce(
                disc, real, fake_d.detach(), spec, patch=patch, for_generator=False
            )
            if not torch.isfinite(loss_d):
                n_skip += 1
                continue
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
            opt_d.step()

            for p in disc.parameters():
                p.requires_grad = False
            fake_list = [wrap.generate(hist, evo).clamp(0.0, SCALE) for _ in range(ensemble_k)]
            loss_adv, _, _ = weighted_patch_bce(
                disc, real, fake_list[0], spec, patch=patch, for_generator=True
            )
            loss_pool = maxpool_mse(real, fake_list)
            loss_g = beta * loss_adv + lambda_pool * loss_pool
            if not torch.isfinite(loss_g):
                n_skip += 1
                opt_g.zero_grad(set_to_none=True)
                continue
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(wrap.gen.parameters(), grad_clip)
            opt_g.step()

            d_f = float(loss_d.detach())
            g_f = float(loss_g.detach())
            adv_f = float(loss_adv.detach())
            pool_f = float(loss_pool.detach())
            run_d += d_f
            run_g += g_f
            run_adv += adv_f
            run_pool += pool_f
            run_n += 1
            pbar.set_postfix(
                {"D": f"{d_f:.3f}", "G": f"{g_f:.1f}", "pool": f"{pool_f:.1f}"},
                refresh=False,
            )
            if step % log_every == 0:
                tqdm.write(
                    f"[train e{epoch} s{step}] D={d_f:.4f} G={g_f:.4f} adv={adv_f:.4f} pool_mse={pool_f:.4f}"
                )
                append_metrics(
                    metrics_path,
                    {
                        "split": "train_step",
                        "epoch": epoch,
                        "step": step,
                        "stage": STAGE,
                        "gan_d": d_f,
                        "gan_g": g_f,
                        "adv": adv_f,
                        "pool_mse": pool_f,
                    },
                )

        val = evaluate_gen(wrap, val_loader, device)
        tr_d = run_d / max(run_n, 1)
        tr_g = run_g / max(run_n, 1)
        tr_adv = run_adv / max(run_n, 1)
        tr_pool = run_pool / max(run_n, 1)
        print(
            f"=== epoch {epoch}: D={tr_d:.4f} G={tr_g:.4f} adv={tr_adv:.4f} pool_mse={tr_pool:.4f} "
            f"val MAE={val['radar_mae128']:.4f} CSI@8={val['csi8']:.4f} "
            f"CSI20={val['csi20']:.4f} CSI30={val['csi30']:.4f} CSI40={val['csi40']:.4f} "
            f"kept={run_n} skip={n_skip} ===",
            flush=True,
        )
        row = {
            "split": "epoch",
            "epoch": epoch,
            "stage": STAGE,
            "gan_d": tr_d,
            "gan_g": tr_g,
            "adv": tr_adv,
            "pool_mse": tr_pool,
            "val_radar_mae128": val["radar_mae128"],
            "val_csi8": val["csi8"],
            "val_csi20": val["csi20"],
            "val_csi30": val["csi30"],
            "val_csi40": val["csi40"],
            "n_batches": run_n,
            "n_skipped": n_skip,
        }
        append_metrics(metrics_path, row)
        extra = {
            "val_radar_mae128": val["radar_mae128"],
            "val_csi8": val["csi8"],
            "val_csi20": val["csi20"],
            "val_csi30": val["csi30"],
            "val_csi40": val["csi40"],
            "flow_ckpt": str(flow_ckpt),
        }
        ckpt = pack_ckpt(wrap, disc, opt_g, opt_d, epoch, extra)
        epoch_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
        torch.save(ckpt, epoch_path)
        torch.save(ckpt, ckpt_dir / "last.pt")
        print(f"  saved {epoch_path.name}", flush=True)
        c40 = float(val["csi40"])
        if math.isfinite(c40) and c40 > best_csi40:
            best_csi40 = c40
            torch.save(ckpt, ckpt_dir / "best_csi40.pt")
            print(f"  saved best_csi40.pt CSI40={c40:.6f}", flush=True)

    print(f"done. best val CSI40={best_csi40:.6f} -> {ckpt_dir / 'best_csi40.pt'}", flush=True)


if __name__ == "__main__":
    main()
