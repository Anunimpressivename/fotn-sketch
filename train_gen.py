"""Train a NowcastNet generator on frozen flow32. Loss is 6 * adv + lambda * MSE."""
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
from sketch.gen_loss import adversarial_loss, discriminator_loss, pixel_l2
from sketch.ncwrap import SCALE, SketchFlowGen, build_discriminator, build_flow, build_generator
from sketch.normalize import NormStats

STAGE = "nc_gen"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_rp_flow32_gen.yaml")
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

    disc = build_discriminator(cfg).to(device)
    gen_lr = float(tcfg.get("gen_lr", 3e-5))
    opt_g = torch.optim.Adam(wrap.gen.parameters(), lr=gen_lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=gen_lr)
    beta = float(tcfg.get("beta_adv", 6.0))
    lambda_l2 = float(tcfg.get("lambda_l2", 0.1))
    ensemble_k = max(int(tcfg.get("ensemble_k", 4)), 1)
    print(
        f"D params={sum(p.numel() for p in disc.parameters())} lr={gen_lr:g} "
        f"beta={beta:g} lambda_l2={lambda_l2:g} K={ensemble_k}",
        flush=True,
    )

    epochs = int(tcfg.get("epochs", 12))
    patience = max(int(tcfg.get("patience", 5)), 1)
    log_every = int(tcfg.get("log_every", 20))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    best_csi = -1.0
    bad = 0
    wrap.train()
    wrap.freeze_flow()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"epoch {epoch} [{STAGE}]")
        run_d = run_g = run_adv = run_l2 = 0.0
        run_n = n_skip = 0
        for step, batch in enumerate(pbar, 1):
            batch = move(batch, device)
            hist, evo, _fields, _traj = wrap.evolve(batch)
            real = batch["radar_gt"].clamp(0.0, 1.0) * SCALE

            wrap.gen.eval()
            with torch.no_grad():
                fake_d = wrap.generate(hist, evo).clamp(0.0, SCALE)
            wrap.gen.train()
            disc.train()
            for p in disc.parameters():
                p.requires_grad = True
            loss_d = discriminator_loss(disc(real), disc(fake_d.detach()))
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
            loss_adv = adversarial_loss(disc(fake_list[0]))
            loss_l2 = pixel_l2(real, fake_list)
            loss_g = beta * loss_adv + lambda_l2 * loss_l2
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
            l2_f = float(loss_l2.detach())
            run_d += d_f
            run_g += g_f
            run_adv += adv_f
            run_l2 += l2_f
            run_n += 1
            pbar.set_postfix({"D": f"{d_f:.3f}", "G": f"{g_f:.1f}", "l2": f"{l2_f:.1f}"}, refresh=False)
            if step % log_every == 0:
                tqdm.write(
                    f"[train e{epoch} s{step}] D={d_f:.4f} G={g_f:.4f} adv={adv_f:.4f} l2={l2_f:.4f}"
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
                        "l2": l2_f,
                    },
                )

        val = evaluate_gen(wrap, val_loader, device)
        tr_d = run_d / max(run_n, 1)
        tr_g = run_g / max(run_n, 1)
        tr_adv = run_adv / max(run_n, 1)
        tr_l2 = run_l2 / max(run_n, 1)
        print(
            f"=== epoch {epoch}: D={tr_d:.4f} G={tr_g:.4f} adv={tr_adv:.4f} l2={tr_l2:.4f} "
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
            "l2": tr_l2,
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
            "flow_ckpt": str(flow_ckpt),
        }
        ckpt = pack_ckpt(wrap, disc, opt_g, opt_d, epoch, extra)
        torch.save(ckpt, ckpt_dir / "last.pt")
        va_csi = float(val["csi8"])
        if math.isfinite(va_csi) and va_csi > best_csi:
            best_csi = va_csi
            bad = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  saved best.pt CSI@8={va_csi:.6f} MAE={val['radar_mae128']:.4f}", flush=True)
        else:
            bad += 1
            print(f"  early-stop {bad}/{patience} best CSI@8={best_csi:.6f}", flush=True)
            if bad >= patience:
                print(f"<<< early stop at epoch={epoch}", flush=True)
                break

    print(f"done. best val CSI@8={best_csi:.6f} -> {ckpt_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
