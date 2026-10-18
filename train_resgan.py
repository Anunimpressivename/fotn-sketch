"""From-scratch flow / conv / res, then a residual GAN stage.

flow, conv, and res use the supervised curriculum. The residual head is
conditioned on the zero-delta evolution plus noise from the res stage onward,
but the discriminator stays off until gan. In gan, res uses a higher learning
rate than flow and conv, and the scalar a is frozen.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from sketch.data import WindowDataset, gather_norm, make_loader
from sketch.losses import residual_lowpass_l1, select_metric, total_loss
from sketch.model import SketchModel
from sketch.normalize import NormStats
from sketch.global_disc import GlobalDiscriminator, global_bce
from sketch.patch_disc import PatchClipDiscriminator, PatchDiscriminator
from sketch.patch_loss import (
    change_pixel_weight,
    maxpool_mse,
    ring_softmax_pool_mse,
    sample_patch_clips,
    sample_patches,
    weighted_clip_bce,
    weighted_patch_bce,
)

SCALE = 128.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def append_metrics(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def save_ckpt(path: Path, model, epoch: int, stage: str, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, "stage": stage, **extra}, path)


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


def patch_kwargs(gcfg: dict) -> dict:
    return {
        "n_frames": int(gcfg.get("patch_frames", 4)),
        "n_biased": int(gcfg.get("patch_biased", 4)),
        "n_uniform": int(gcfg.get("patch_uniform", 4)),
        "patch": int(gcfg.get("patch_size", 16)),
        "temperature": float(gcfg.get("patch_temperature", 32.0)),
    }


def clip_len_of(gcfg: dict) -> int:
    return int(gcfg.get("patch_clip_len", 0) or 0)


def sample_local(real: torch.Tensor, gcfg: dict, pk: dict, patch: int) -> dict:
    if clip_len_of(gcfg) > 1:
        return sample_patch_clips(
            real,
            clip_len=clip_len_of(gcfg),
            n_biased=pk["n_biased"],
            n_uniform=pk["n_uniform"],
            patch=patch,
            temperature=pk["temperature"],
        )
    sample_kw = {k: pk[k] for k in ("n_frames", "n_biased", "n_uniform", "temperature")}
    return sample_patches(real, patch=patch, **sample_kw)


def local_bce(disc, real, fake, spec, gcfg, patch, *, for_generator: bool):
    if clip_len_of(gcfg) > 1:
        return weighted_clip_bce(disc, real, fake, spec, patch=patch, for_generator=for_generator)
    return weighted_patch_bce(disc, real, fake, spec, patch=patch, for_generator=for_generator)


@torch.no_grad()
def run_val(model, loader, device, stage: str, use_amp: bool, cfg: dict | None = None) -> float:
    model.eval()
    scores = []
    amp_dev = "cuda" if device.type == "cuda" else "cpu"
    for batch in loader:
        batch = move(batch, device)
        with torch.amp.autocast(device_type=amp_dev, enabled=use_amp):
            _, traj = model.forward_batch(batch, stage)
        scores.append(select_metric(traj, batch, cfg))
    model.train()
    return float(np.mean(scores)) if scores else float("inf")


@torch.no_grad()
def evaluate_gan(model, loader, device) -> dict:
    model.eval()
    mae_sum = 0.0
    buckets = {8.0: [], 51.2: [], 76.8: [], 102.4: []}
    n = 0
    for batch in loader:
        batch = move(batch, device)
        _fields, traj = model.forward_batch(batch, "gan")
        pred = traj["radar"].clamp(0.0, 1.0) * SCALE
        gt = batch["radar_gt"].clamp(0.0, 1.0) * SCALE
        mae_sum += float((pred - gt).abs().mean())
        for thr in buckets:
            c = _csi(pred, gt, thr)
            if math.isfinite(c):
                buckets[thr].append(c)
        n += 1
    model.train()

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


def run_supervised(model, train_loader, val_loader, cfg, device, ckpt_dir, log_path) -> int:
    tcfg = cfg["train"]
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dev = "cuda" if device.type == "cuda" else "cpu"
    global_ep = 0
    best_all = float("inf")
    la = cfg["loss"]
    if bool(la.get("radar_change_weight", False)):
        dw_mode = str(la.get("change_weight_mode", "abs"))
        print(
            "radar change weight on advect_radar and evo_radar: "
            f"mode={dw_mode} "
            f"floor={float(la.get('change_weight_floor', 1.0)):g} "
            f"max={float(la.get('change_weight_max', 64.0)):g} "
            f"scale={float(la.get('change_weight_scale', 128.0)):g} "
            f"bias={float(la.get('change_weight_delta_bias', 16.0)):g} "
            f"denom={float(la.get('change_weight_denom', 64.0)):g} "
            f"spatial={int(la.get('change_weight_spatial', 1))} "
            f"temporal={int(la.get('change_weight_temporal', 0))} "
            f"normalize={bool(la.get('change_weight_normalize', False))}",
            flush=True,
        )
    for stage_cfg in tcfg["stages"]:
        stage = stage_cfg["name"]
        if stage == "gan":
            continue
        model.configure(stage)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
        best = float("inf")
        bad = 0
        print(f"\n=== stage {stage} epochs={stage_cfg['epochs']} trainable={sum(p.numel() for p in params)} ===", flush=True)
        for local_ep in range(1, int(stage_cfg["epochs"]) + 1):
            global_ep += 1
            model.train()
            model.configure(stage)
            t0 = time.time()
            run_logs = []
            for batch in tqdm(train_loader, desc=f"{stage} e{local_ep}", leave=False):
                batch = move(batch, device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=amp_dev, enabled=use_amp):
                    fields, traj = model.forward_batch(batch, stage)
                    loss, logs = total_loss(traj, fields, batch, stage, cfg)
                if not torch.isfinite(loss):
                    print("skip non-finite loss", flush=True)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, float(tcfg["grad_clip"]))
                scaler.step(opt)
                scaler.update()
                run_logs.append(logs)
            va = run_val(model, val_loader, device, stage, use_amp, cfg)
            mean_tr = {k: float(np.mean([x[k] for x in run_logs if k in x])) for k in run_logs[0]} if run_logs else {}
            row = {
                "epoch": global_ep, "stage": stage, "local": local_ep,
                "va_rp": va, "a": float(model.a.detach()),
                "sec": time.time() - t0, **{f"tr_{k}": v for k, v in mean_tr.items()},
            }
            append_metrics(log_path, row)
            print(
                f"e{global_ep} {stage}/{local_ep} va_rp={va:.4f} "
                f"tr={mean_tr.get('loss', float('nan')):.4f} a={row['a']:.3f}",
                flush=True,
            )
            save_ckpt(ckpt_dir / "last.pt", model, global_ep, stage, {"va_rp": va})
            if va < best - 1e-5:
                best = va
                bad = 0
                save_ckpt(ckpt_dir / f"best_stage_{stage}.pt", model, global_ep, stage, {"va_rp": va})
            else:
                bad += 1
            if va < best_all - 1e-5:
                best_all = va
                save_ckpt(ckpt_dir / "best.pt", model, global_ep, stage, {"va_rp": va})
            if bad >= int(stage_cfg.get("patience", 99)):
                print(f"early stop {stage} after {local_ep} (best va_rp={best:.4f})", flush=True)
                break
        sp = ckpt_dir / f"best_stage_{stage}.pt"
        if sp.exists():
            state = torch.load(sp, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            print(f"reload {sp.name} va_rp={state.get('va_rp')}", flush=True)
    return global_ep


def pretrain_discriminator(model, disc, opt_d, loader, device, gcfg, metrics_path, grad_clip: float, global_disc=None) -> None:
    max_steps = int(gcfg.get("pretrain_max_steps", 0))
    stop_d = float(gcfg.get("pretrain_stop_d", 0.9))
    min_gap = float(gcfg.get("pretrain_min_gap", 1.0))
    min_steps = int(gcfg.get("pretrain_min_steps", 100))
    log_every = int(gcfg.get("log_every", 20))
    pk = patch_kwargs(gcfg)
    patch = pk.pop("patch")
    disc.train()
    model.eval()
    recent_d: list[float] = []
    recent_gap: list[float] = []
    print(
        f"pretrain D on zero-delta evolution vs GT, stop when D<{stop_d:g} and logit gap>{min_gap:g} "
        f"after {min_steps} steps",
        flush=True,
    )
    step = 0
    for batch in tqdm(loader, desc="pretrain D"):
        step += 1
        if max_steps and step > max_steps:
            break
        batch = move(batch, device)
        with torch.no_grad():
            _fields, traj = model.forward_batch(batch, "conv")
            evo = traj["radar"].clamp(0.0, 1.0) * SCALE
        real = batch["radar_gt"].clamp(0.0, 1.0) * SCALE
        spec = sample_local(real, gcfg, pk, patch)
        loss_patch, real_logit, fake_logit = local_bce(
            disc, real, evo, spec, gcfg, patch, for_generator=False
        )
        loss_d = loss_patch
        if global_disc is not None:
            loss_gd, _, _ = global_bce(global_disc, real, evo, spec["times"], for_generator=False)
            loss_d = loss_patch + loss_gd
        if not torch.isfinite(loss_d):
            continue
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
        if global_disc is not None:
            torch.nn.utils.clip_grad_norm_(global_disc.parameters(), grad_clip)
        opt_d.step()
        d_f = float(loss_patch.detach())
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
            append_metrics(metrics_path, {
                "split": "pretrain_step", "step": step, "stage": "gan",
                "gan_d": d_f, "real_logit": float(real_logit), "fake_logit": float(fake_logit), "gap": gap,
            })
            if step >= min_steps and md < stop_d and mg > min_gap:
                print(f"pretrain stop at step {step}: meanD={md:.4f} gap={mg:.3f}", flush=True)
                break
    model.train()


def run_gan(model, disc, train_loader, val_loader, cfg, device, ckpt_dir, log_path, global_ep: int, global_disc=None) -> None:
    tcfg = cfg["train"]
    gcfg = tcfg["gan"]
    model.configure("gan")
    if model.a_raw.requires_grad:
        raise RuntimeError("scalar a must be frozen in the gan stage")
    res_params = [p for p in model.res.parameters() if p.requires_grad]
    evo_params = [p for p in list(model.flow.parameters()) + list(model.conv.parameters()) if p.requires_grad]
    opt_g = torch.optim.AdamW(
        [
            {"params": res_params, "lr": float(gcfg["res_lr"])},
            {"params": evo_params, "lr": float(gcfg["evo_lr"])},
        ],
        weight_decay=float(tcfg["weight_decay"]),
    )
    d_params = list(disc.parameters())
    if global_disc is not None:
        d_params += list(global_disc.parameters())
    opt_d = torch.optim.Adam(d_params, lr=float(gcfg["disc_lr"]))
    beta = float(gcfg.get("beta_adv", 6.0))
    lambda_pool = float(gcfg.get("lambda_pool_mse", 0.02))
    lambda_mae = float(gcfg.get("lambda_pixel_mae", 0.0))
    lambda_mse = float(gcfg.get("lambda_pixel_mse", 0.0))
    lambda_mae_sq = float(gcfg.get("lambda_mae_sq", 0.0))
    lambda_global = float(gcfg.get("lambda_global_adv", 0.0))
    lambda_res_low = float(gcfg.get("lambda_res_low", 0.0))
    res_lowpass_k = int(gcfg.get("res_lowpass_k", 5))
    res_highpass_w = float(gcfg.get("res_highpass_weight", 0.0))
    pool_kernel = int(gcfg.get("pool_kernel_size", 5))
    pool_stride = int(gcfg.get("pool_stride", 2))
    pool_mode = str(gcfg.get("pool_mode", "max")).strip().lower()
    pool_tau = float(gcfg.get("pool_tau", 16.0))
    pool_metric = str(gcfg.get("pool_metric", "chebyshev")).strip().lower()
    pixel_weight = str(gcfg.get("pixel_weight", "none")).strip().lower()
    lcfg = cfg["loss"]
    dw_floor = float(gcfg.get("change_weight_floor", lcfg.get("change_weight_floor", 1.0)))
    dw_max = float(gcfg.get("change_weight_max", lcfg.get("change_weight_max", 64.0)))
    dw_mode = str(gcfg.get("change_weight_mode", lcfg.get("change_weight_mode", "abs")))
    dw_bias = float(gcfg.get("change_weight_delta_bias", lcfg.get("change_weight_delta_bias", 16.0)))
    dw_denom = float(gcfg.get("change_weight_denom", lcfg.get("change_weight_denom", 64.0)))
    dw_spatial = int(gcfg.get("change_weight_spatial", lcfg.get("change_weight_spatial", 1)))
    dw_temporal = int(gcfg.get("change_weight_temporal", lcfg.get("change_weight_temporal", 0)))
    dw_norm = bool(gcfg.get("change_weight_normalize", lcfg.get("change_weight_normalize", False)))
    pk = patch_kwargs(gcfg)
    patch = pk["patch"]
    grad_clip = float(tcfg["grad_clip"])
    print(
        f"\n=== stage gan epochs={gcfg['epochs']} res_params={sum(p.numel() for p in res_params)} "
        f"evo_params={sum(p.numel() for p in evo_params)} disc={sum(p.numel() for p in disc.parameters())} "
        f"res_lr={gcfg['res_lr']} evo_lr={gcfg['evo_lr']} beta={beta:g} pool={lambda_pool:g} "
        f"pool_mode={pool_mode} pool_k={pool_kernel} pool_s={pool_stride} pool_tau={pool_tau:g} "
        f"pool_metric={pool_metric} "
        f"pixel_mae={lambda_mae:g} pixel_mse={lambda_mse:g} mae_sq={lambda_mae_sq:g} "
        f"res_low={lambda_res_low:g} res_k={res_lowpass_k} res_hp={res_highpass_w:g} "
        f"detach_phys={model.detach_phys_cond} pixel_weight={pixel_weight} "
        f"dw_mode={dw_mode} dw_floor={dw_floor:g} dw_max={dw_max:g} "
        f"dw_bias={dw_bias:g} dw_denom={dw_denom:g} dw_scale=1 "
        f"dw_spatial={dw_spatial} dw_temporal={dw_temporal} dw_norm={dw_norm} "
        f"global={lambda_global:g} patch_size={patch} clip_len={clip_len_of(gcfg)} "
        f"global_disc={0 if global_disc is None else sum(p.numel() for p in global_disc.parameters())} "
        f"supervised=on ===",
        flush=True,
    )
    pretrain_discriminator(
        model, disc, opt_d, train_loader, device, gcfg, log_path, grad_clip, global_disc=global_disc
    )
    epochs = int(gcfg["epochs"])
    log_every = int(gcfg.get("log_every", 20))
    best_csi40 = -1.0
    checked_grad = False
    for local_ep in range(1, epochs + 1):
        global_ep += 1
        model.train()
        model.configure("gan")
        disc.train()
        if global_disc is not None:
            global_disc.train()
        t0 = time.time()
        run_d = run_g = run_adv = run_pool = run_mae = run_mse = run_mae_sq = run_sup = run_gadv = run_res = 0.0
        run_pool_ent = run_dw = 0.0
        run_dw_n = 0
        run_n = n_skip = 0
        pbar = tqdm(train_loader, desc=f"gan e{local_ep}", leave=False)
        for step, batch in enumerate(pbar, 1):
            batch = move(batch, device)
            fields, traj = model.forward_batch(batch, "gan")
            real = batch["radar_gt"].clamp(0.0, 1.0) * SCALE
            fake = traj["radar"].clamp(0.0, 1.0) * SCALE
            spec = sample_local(real, gcfg, pk, patch)

            for p in disc.parameters():
                p.requires_grad = True
            if global_disc is not None:
                for p in global_disc.parameters():
                    p.requires_grad = True
            loss_d, real_logit, fake_logit = local_bce(
                disc, real, fake.detach(), spec, gcfg, patch, for_generator=False
            )
            if global_disc is not None:
                loss_gd, _, _ = global_bce(
                    global_disc, real, fake.detach(), spec["times"], for_generator=False
                )
                loss_d = loss_d + loss_gd
            if not torch.isfinite(loss_d):
                n_skip += 1
                continue
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), grad_clip)
            if global_disc is not None:
                torch.nn.utils.clip_grad_norm_(global_disc.parameters(), grad_clip)
            opt_d.step()

            for p in disc.parameters():
                p.requires_grad = False
            if global_disc is not None:
                for p in global_disc.parameters():
                    p.requires_grad = False
            loss_adv, _, _ = local_bce(
                disc, real, fake, spec, gcfg, patch, for_generator=True
            )
            if global_disc is None:
                loss_gadv = fake.new_zeros(())
            else:
                loss_gadv, _, _ = global_bce(
                    global_disc, real, fake, spec["times"], for_generator=True
                )
            if pool_mode in ("ring_softmax", "ring"):
                loss_pool, pool_ent = ring_softmax_pool_mse(
                    real, [fake],
                    kernel_size=pool_kernel, stride=pool_stride, tau=pool_tau,
                    metric=pool_metric,
                )
            else:
                loss_pool = maxpool_mse(
                    real, [fake], kernel_size=pool_kernel, stride=pool_stride
                )
                pool_ent = loss_pool.new_zeros(())
            err = (fake - real).abs()
            if pixel_weight == "change":
                hist_last = batch["radar_hist"][:, -1].clamp(0.0, 1.0) * SCALE
                w_pix = change_pixel_weight(
                    real, hist_last,
                    floor=dw_floor, w_max=dw_max, scale=1.0,
                    mode=dw_mode, delta_bias=dw_bias, denom=dw_denom,
                    spatial_kernel=dw_spatial, temporal_radius=dw_temporal,
                    normalize_mean=dw_norm,
                )
                loss_mae = (err * w_pix).mean()
            elif pixel_weight in ("none", ""):
                w_pix = None
                loss_mae = err.mean()
            else:
                raise ValueError(f"unknown pixel_weight {pixel_weight}")
            loss_mse = (fake - real).square().mean()
            loss_mae_sq = loss_mae.square()
            loss_sup, _sup_logs = total_loss(traj, fields, batch, "gan", cfg)
            if lambda_res_low > 0:
                loss_res, _, _ = residual_lowpass_l1(
                    fields["delta_radar"],
                    fields["delta_pwv"],
                    kernel=res_lowpass_k,
                    highpass_weight=res_highpass_w,
                )
            else:
                loss_res = fake.new_zeros(())
            loss_g = (
                beta * loss_adv
                + lambda_global * loss_gadv
                + lambda_pool * loss_pool
                + lambda_mae * loss_mae
                + lambda_mse * loss_mse
                + lambda_mae_sq * loss_mae_sq
                + lambda_res_low * loss_res
                + loss_sup
            )
            if not torch.isfinite(loss_g):
                n_skip += 1
                opt_g.zero_grad(set_to_none=True)
                continue
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if not checked_grad:
                flow_g = model.flow.enc0.net[0].weight.grad
                conv_g = model.conv.enc0.net[0].weight.grad
                flow_n = 0.0 if flow_g is None else float(flow_g.detach().norm())
                conv_n = 0.0 if conv_g is None else float(conv_g.detach().norm())
                print(
                    f"gan grad check flow={flow_n:.3e} conv={conv_n:.3e} "
                    f"a_grad={model.a_raw.grad is not None}",
                    flush=True,
                )
                checked_grad = True
            torch.nn.utils.clip_grad_norm_(res_params + evo_params, grad_clip)
            opt_g.step()

            d_f = float(loss_d.detach())
            g_f = float(loss_g.detach())
            adv_f = float(loss_adv.detach())
            gadv_f = float(loss_gadv.detach())
            pool_f = float(loss_pool.detach())
            pool_ent_f = float(pool_ent.detach())
            mae_f = float(loss_mae.detach())
            dw_f = float("nan") if w_pix is None else float(w_pix.detach().mean())
            mse_f = float(loss_mse.detach())
            mae_sq_f = float(loss_mae_sq.detach())
            sup_f = float(loss_sup.detach())
            res_f = float(loss_res.detach())
            run_d += d_f
            run_g += g_f
            run_adv += adv_f
            run_gadv += gadv_f
            run_pool += pool_f
            run_pool_ent += pool_ent_f
            run_mae += mae_f
            if w_pix is not None:
                run_dw += dw_f
                run_dw_n += 1
            run_mse += mse_f
            run_mae_sq += mae_sq_f
            run_sup += sup_f
            run_res += res_f
            run_n += 1
            pbar.set_postfix(
                {
                    "D": f"{d_f:.3f}",
                    "G": f"{g_f:.2f}",
                    "mae": f"{mae_f:.2f}",
                    "mae2": f"{mae_sq_f:.2f}",
                },
                refresh=False,
            )
            if step % log_every == 0:
                tqdm.write(
                    f"[gan e{local_ep} s{step}] D={d_f:.4f} G={g_f:.4f} adv={adv_f:.4f} "
                    f"gadv={gadv_f:.4f} pool_mse={pool_f:.4f} pool_ent={pool_ent_f:.3f} "
                    f"pixel_mae={mae_f:.4f} dw_mean={dw_f:.3f} pixel_mse={mse_f:.4f} mae_sq={mae_sq_f:.4f} "
                    f"res_low={res_f:.4f} sup={sup_f:.4f}"
                )
                append_metrics(log_path, {
                    "split": "train_step", "epoch": global_ep, "local": local_ep, "step": step,
                    "stage": "gan", "gan_d": d_f, "gan_g": g_f, "adv": adv_f, "gadv": gadv_f,
                    "pool_mse": pool_f, "pool_ent": pool_ent_f, "pixel_mae": mae_f, "dw_mean": dw_f,
                    "pixel_mse": mse_f,
                    "mae_sq": mae_sq_f, "res_low": res_f, "sup": sup_f,
                })

        val = evaluate_gan(model, val_loader, device)
        tr_d = run_d / max(run_n, 1)
        tr_g = run_g / max(run_n, 1)
        tr_adv = run_adv / max(run_n, 1)
        tr_gadv = run_gadv / max(run_n, 1)
        tr_pool = run_pool / max(run_n, 1)
        tr_pool_ent = run_pool_ent / max(run_n, 1)
        tr_mae = run_mae / max(run_n, 1)
        tr_dw = float("nan") if run_dw_n == 0 else run_dw / run_dw_n
        tr_mse = run_mse / max(run_n, 1)
        tr_mae_sq = run_mae_sq / max(run_n, 1)
        tr_sup = run_sup / max(run_n, 1)
        tr_res = run_res / max(run_n, 1)
        print(
            f"e{global_ep} gan/{local_ep} D={tr_d:.4f} G={tr_g:.4f} adv={tr_adv:.4f} gadv={tr_gadv:.4f} "
            f"pool_mse={tr_pool:.4f} pool_ent={tr_pool_ent:.3f} pixel_mae={tr_mae:.4f} dw_mean={tr_dw:.3f} pixel_mse={tr_mse:.4f} "
            f"mae_sq={tr_mae_sq:.4f} res_low={tr_res:.4f} sup={tr_sup:.4f} "
            f"val MAE={val['radar_mae128']:.4f} CSI@8={val['csi8']:.4f} "
            f"CSI20={val['csi20']:.4f} CSI30={val['csi30']:.4f} CSI40={val['csi40']:.4f} "
            f"a={float(model.a.detach()):.3f} sec={time.time() - t0:.0f}",
            flush=True,
        )
        row = {
            "epoch": global_ep, "stage": "gan", "local": local_ep,
            "gan_d": tr_d, "gan_g": tr_g, "adv": tr_adv, "gadv": tr_gadv,
            "pool_mse": tr_pool, "pool_ent": tr_pool_ent,
            "pixel_mae": tr_mae, "dw_mean": tr_dw, "pixel_mse": tr_mse, "mae_sq": tr_mae_sq,
            "res_low": tr_res, "sup": tr_sup,
            "val_radar_mae128": val["radar_mae128"],
            "val_csi8": val["csi8"], "val_csi20": val["csi20"],
            "val_csi30": val["csi30"], "val_csi40": val["csi40"],
            "a": float(model.a.detach()), "sec": time.time() - t0,
            "n_batches": run_n, "n_skipped": n_skip,
        }
        append_metrics(log_path, row)
        extra = {
            "val_radar_mae128": val["radar_mae128"],
            "val_csi8": val["csi8"], "val_csi20": val["csi20"],
            "val_csi30": val["csi30"], "val_csi40": val["csi40"],
            "disc": disc.state_dict(),
            "disc_global": None if global_disc is None else global_disc.state_dict(),
        }
        save_ckpt(ckpt_dir / f"epoch_{local_ep:02d}.pt", model, global_ep, "gan", extra)
        save_ckpt(ckpt_dir / "last.pt", model, global_ep, "gan", extra)
        c40 = float(val["csi40"])
        if math.isfinite(c40) and c40 > best_csi40:
            best_csi40 = c40
            save_ckpt(ckpt_dir / "best_csi40.pt", model, global_ep, "gan", extra)
            print(f"  saved best_csi40.pt CSI40={c40:.6f}", flush=True)
    print(f"done. best val CSI40={best_csi40:.6f} -> {ckpt_dir / 'best_csi40.pt'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config_rp_flow32_resgan_gdisc_g025_ring16.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    use_precip = bool(dcfg.get("use_precip", True))
    ckpt_dir = Path(tcfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "metrics.jsonl"

    norm_path = ckpt_dir / "norm_stats.json"
    if norm_path.exists():
        norm = NormStats.load(norm_path)
    else:
        print("computing norm stats on train hist...", flush=True)
        raw = WindowDataset(
            dcfg["train_root"], dcfg["train_months"],
            dcfg["t_in"], dcfg["t_out"], tuple(dcfg["image_hw"]),
            use_precip=use_precip,
        )
        norm = gather_norm(raw, max_windows=256)
        norm.save(norm_path)
        print(norm, flush=True)

    train_loader = make_loader(
        dcfg["train_root"], dcfg["train_months"], norm,
        dcfg["t_in"], dcfg["t_out"], dcfg["image_hw"],
        dcfg["batch_size"], dcfg["num_workers"], True,
        use_precip=use_precip,
    )
    val_loader = make_loader(
        dcfg["eval_root"], dcfg["val_months"], norm,
        dcfg["t_in"], dcfg["t_out"], dcfg["image_hw"],
        dcfg["batch_size"], dcfg["num_workers"], False,
        use_precip=use_precip,
    )
    print(f"train windows={len(train_loader.dataset)} val windows={len(val_loader.dataset)} use_precip={use_precip}", flush=True)

    model = SketchModel(
        t_in=dcfg["t_in"], t_out=dcfg["t_out"], base_ch=mcfg["base_ch"],
        flow_base_ch=mcfg.get("flow_base_ch"),
        a_init=mcfg["a_init"], u_scale=mcfg["u_scale"], delta_scale=mcfg["delta_scale"],
        clamp_state=mcfg["clamp_state"], clamp_s=mcfg["clamp_s"],
        use_precip=use_precip,
        residual_cond=bool(mcfg.get("residual_cond", False)),
        detach_phys_cond=bool(mcfg.get("detach_phys_cond", False)),
        res_base_ch=mcfg.get("res_base_ch"),
        n_noise=int(mcfg.get("n_noise", 4)),
    ).to(device)
    model.radar_mean.fill_(float(norm.radar_mean))
    model.radar_std.fill_(float(norm.radar_std))
    nparam = sum(p.numel() for p in model.parameters())
    nflow = sum(p.numel() for p in model.flow.parameters())
    nres = sum(p.numel() for p in model.res.parameters())
    print(
        f"params={nparam} flow_params={nflow} res_params={nres} "
        f"res_in={model.res.enc0.net[0].in_channels} res_base={model.res_base_ch}",
        flush=True,
    )
    init_ckpt = tcfg.get("init_ckpt")
    if init_ckpt:
        state = torch.load(init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(
            f"loaded {init_ckpt} epoch={state.get('epoch')} stage={state.get('stage')} "
            f"va_rp={state.get('va_rp')}",
            flush=True,
        )
    if bool(tcfg.get("skip_curriculum", False)):
        global_ep = 0
        print("skip flow/conv/res, gan only", flush=True)
    else:
        global_ep = run_supervised(model, train_loader, val_loader, cfg, device, ckpt_dir, log_path)
    gcfg = tcfg["gan"]
    clip_len = clip_len_of(gcfg)
    if clip_len > 1:
        disc = PatchClipDiscriminator(
            base_c=int(gcfg.get("patch_clip_disc_base_c", 16)),
            clip_len=clip_len,
        ).to(device)
        print(
            f"clip disc params={sum(p.numel() for p in disc.parameters())} clip_len={clip_len}",
            flush=True,
        )
    else:
        disc = PatchDiscriminator(
            base_c=int(gcfg.get("patch_disc_base_c", 32)),
            patch_size=int(gcfg.get("patch_size", 16)),
        ).to(device)
        print(
            f"patch disc params={sum(p.numel() for p in disc.parameters())} "
            f"patch_size={int(gcfg.get('patch_size', 16))}",
            flush=True,
        )
    global_disc = None
    if float(gcfg.get("lambda_global_adv", 0.0)) > 0:
        global_disc = GlobalDiscriminator(base_c=int(gcfg.get("global_disc_base_c", 16))).to(device)
        print(f"global disc params={sum(p.numel() for p in global_disc.parameters())}", flush=True)
    run_gan(
        model, disc, train_loader, val_loader, cfg, device, ckpt_dir, log_path, global_ep,
        global_disc=global_disc,
    )


if __name__ == "__main__":
    main()
