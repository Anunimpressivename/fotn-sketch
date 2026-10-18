from __future__ import annotations

import argparse
import json
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
from sketch.losses import select_metric, total_loss
from sketch.model import SketchModel
from sketch.normalize import NormStats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def run_val(model, loader, device, stage: str, use_amp: bool) -> float:
    model.eval()
    scores = []
    amp_dev = "cuda" if device.type == "cuda" else "cpu"
    for batch in loader:
        batch = move(batch, device)
        with torch.amp.autocast(device_type=amp_dev, enabled=use_amp):
            _, traj = model.forward_batch(batch, stage)
        scores.append(select_metric(traj, batch))
    model.train()
    return float(np.mean(scores)) if scores else float("inf")


def save_ckpt(path: Path, model, epoch: int, stage: str, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, "stage": stage, **extra}, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
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
        print("computing norm stats on train hist...")
        raw = WindowDataset(
            dcfg["train_root"], dcfg["train_months"],
            dcfg["t_in"], dcfg["t_out"], tuple(dcfg["image_hw"]),
            use_precip=use_precip,
        )
        norm = gather_norm(raw, max_windows=256)
        norm.save(norm_path)
        print(norm)

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
    print(f"train windows={len(train_loader.dataset)} val windows={len(val_loader.dataset)} use_precip={use_precip}")

    model = SketchModel(
        t_in=dcfg["t_in"], t_out=dcfg["t_out"], base_ch=mcfg["base_ch"],
        flow_base_ch=mcfg.get("flow_base_ch"),
        a_init=mcfg["a_init"], u_scale=mcfg["u_scale"], delta_scale=mcfg["delta_scale"],
        clamp_state=mcfg["clamp_state"], clamp_s=mcfg["clamp_s"],
        use_precip=use_precip,
    ).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    nflow = sum(p.numel() for p in model.flow.parameters())
    print(
        f"params={nparam} flow_params={nflow} flow_base_ch={model.flow_base_ch} "
        f"base_ch={mcfg['base_ch']} conv_in={model.conv.enc0.net[0].in_channels}"
    )

    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dev = "cuda" if device.type == "cuda" else "cpu"
    global_ep = 0
    best_all = float("inf")

    for stage_cfg in tcfg["stages"]:
        stage = stage_cfg["name"]
        model.configure(stage)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
        best = float("inf")
        bad = 0
        print(f"\n=== stage {stage} epochs={stage_cfg['epochs']} ===")
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
                    print("skip non-finite loss")
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, float(tcfg["grad_clip"]))
                scaler.step(opt)
                scaler.update()
                run_logs.append(logs)
            va = run_val(model, val_loader, device, stage, use_amp)
            mean_tr = {k: float(np.mean([x[k] for x in run_logs if k in x])) for k in run_logs[0]} if run_logs else {}
            row = {
                "epoch": global_ep, "stage": stage, "local": local_ep,
                "va_rp": va, "a": float(model.a.detach()),
                "sec": time.time() - t0, **{f"tr_{k}": v for k, v in mean_tr.items()},
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"e{global_ep} {stage}/{local_ep} va_rp={va:.4f} "
                f"tr={mean_tr.get('loss', float('nan')):.4f} a={row['a']:.3f}"
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
                print(f"early stop {stage} after {local_ep} (best va_rp={best:.4f})")
                break
        # reload stage best before next stage
        sp = ckpt_dir / f"best_stage_{stage}.pt"
        if sp.exists():
            state = torch.load(sp, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            print(f"reload {sp.name} va_rp={state.get('va_rp')}")

    print(f"done. best va_rp={best_all:.4f} -> {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
