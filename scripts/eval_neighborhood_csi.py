"""Neighborhood CSI at 20/30/40 dBZ for ring16, ConvLSTM, and NowcastNet.

The neighborhood is the one already used for NowcastNet validation: max-pool
the intensity with a 5x5 window and stride 2, then score pixel CSI. Thresholds
stay on the 0-128 scale (51.2 / 76.8 / 102.4). Aggregation matches the existing
point-CSI tests: mean of per-batch scores, and hourly scores are the mean of
the first 10 leads, the first 20 leads, and all 30 leads.

Checkpoints are the ones used in the point-CSI comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TH = ((51.2, "20"), (76.8, "30"), (102.4, "40"))
KERNEL = 5
STRIDE = 2


def pool_intensity(x: np.ndarray) -> np.ndarray:
    lead = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(-1, 1, x.shape[-2], x.shape[-1])
    pooled = F.max_pool2d(lead, kernel_size=KERNEL, stride=STRIDE)
    return pooled[:, 0].numpy().reshape(*x.shape[:-2], pooled.shape[-2], pooled.shape[-1])


def csi(pred: np.ndarray, gt: np.ndarray, thr: float) -> float:
    pb = pred >= thr
    tb = gt >= thr
    hits = int(np.logical_and(pb, tb).sum())
    miss = int(np.logical_and(tb, ~pb).sum())
    fa = int(np.logical_and(pb, ~tb).sum())
    denom = hits + miss + fa
    return hits / denom if denom else float("nan")


def nanmean(xs) -> float:
    a = np.array(xs, dtype=np.float64)
    return float(np.nanmean(a)) if a.size else float("nan")


def summarize(per_batch: dict, per_lead: dict, num_samples: int) -> dict:
    out = {
        "num_samples": num_samples,
        "neighborhood": {"op": "max_pool2d", "kernel": KERNEL, "stride": STRIDE, "padding": 0, "on": "intensity"},
        "thresholds": {"20": 51.2, "30": 76.8, "40": 102.4},
        "csi": {},
        "csi_dbz_hour": {},
        "csi_by_lead": {},
    }
    for thr, name in TH:
        series = [nanmean(per_lead[thr][i]) for i in range(30)]
        out["csi"][name] = nanmean(per_batch[thr])
        out["csi_by_lead"][name] = series
        out["csi_dbz_hour"][name] = {
            "1h": float(np.mean(series[:10])),
            "2h": float(np.mean(series[:20])),
            "3h": float(np.mean(series)),
        }
    return out


def accumulate(pred: np.ndarray, gt: np.ndarray, per_batch: dict, per_lead: dict) -> None:
    pooled_p = pool_intensity(pred)
    pooled_g = pool_intensity(gt)
    for thr, _name in TH:
        per_batch[thr].append(csi(pooled_p, pooled_g, thr))
        for li in range(pred.shape[1]):
            per_lead[thr][li].append(csi(pooled_p[:, li], pooled_g[:, li], thr))


def fresh():
    per_batch = {thr: [] for thr, _ in TH}
    per_lead = {thr: [[] for _ in range(30)] for thr, _ in TH}
    return per_batch, per_lead


def run_ring16() -> None:
    root = Path("/root/autodl-fs/fotn-sketch1")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    import yaml
    from eval_resgan_g025_ring16_ep4_test import build
    from sketch.data import make_loader
    from sketch.normalize import NormStats

    run = root / "runs" / "ring16"
    cfg = yaml.safe_load((root / "configs/config_rp_flow32_resgan_gdisc_g025_ring16.yaml").read_text(encoding="utf-8"))
    dcfg = cfg["data"]
    device = torch.device("cuda")
    norm = NormStats.load(run / "norm_stats.json")
    loader = make_loader(
        dcfg["eval_root"], dcfg["test_months"], norm,
        dcfg["t_in"], dcfg["t_out"], tuple(dcfg["image_hw"]),
        dcfg["batch_size"], dcfg["num_workers"], False, use_precip=False,
    )
    ckpt = run / "best_csi40.pt"
    model = build(cfg, ckpt, norm, device)
    model.eval()
    per_batch, per_lead = fresh()
    print(f"ring16 samples={len(loader.dataset)} batches={len(loader)}", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            torch.manual_seed(1000 + i)
            _fields, traj = model.forward_batch(batch, "gan")
            pred = (traj["radar"].clamp(0, 1) * 128.0).cpu().numpy()
            gt = (batch["radar_gt"].clamp(0, 1) * 128.0).cpu().numpy()
            accumulate(pred, gt, per_batch, per_lead)
            if (i + 1) % 100 == 0:
                print(f"  ring16 {i + 1}/{len(loader)}", flush=True)
    metrics = summarize(per_batch, per_lead, len(loader.dataset))
    metrics.update({"model": "ring16", "ckpt": ckpt.name, "epoch": 4})
    dest = run / "eval_metrics_test_epoch04_csin.json"
    dest.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("ring16", {k: metrics["csi"][k] for k in ("20", "30", "40")}, metrics["csi_dbz_hour"]["40"], flush=True)
    print("wrote", dest, flush=True)


def run_convlstm() -> None:
    root = Path("/root/autodl-fs/baselines/convlstm")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "OpenSTL"))
    from torch.utils.data import DataLoader
    from eval_radar_csi import build_model, predict_future
    from openstl.datasets.dataloader_radar_pwv import TEST_MONTHS, RadarPwvDataset, resolve_roots

    ckpt_path = root / "work_dirs/radar_pwv_convlstm_csi20_lr1e4/checkpoints/best.ckpt"
    device = torch.device("cuda")
    _, eval_root = resolve_roots("/root/autodl-tmp")
    dataset = RadarPwvDataset(eval_root, TEST_MONTHS, use_augment=False)
    loader = DataLoader(dataset, batch_size=12, shuffle=False, num_workers=4, pin_memory=True)
    model = build_model(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    per_batch, per_lead = fresh()
    print(f"convlstm samples={len(dataset)} batches={len(loader)} epoch={int(ckpt['epoch'])}", flush=True)
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            pred = predict_future(model, batch_x, batch_y)
            pred_r = (pred[:, :, 0].clamp(0, 1) * 128.0).cpu().numpy()
            true_r = (batch_y[:, :, 0].clamp(0, 1) * 128.0).cpu().numpy()
            accumulate(pred_r, true_r, per_batch, per_lead)
            if (i + 1) % 50 == 0:
                print(f"  convlstm {i + 1}/{len(loader)}", flush=True)
    metrics = summarize(per_batch, per_lead, len(dataset))
    metrics.update({"model": "convlstm", "ckpt": ckpt_path.name, "epoch": int(ckpt["epoch"]), "lr": 1e-4})
    dest = root / "work_dirs/radar_pwv_convlstm_csi20_lr1e4/eval_metrics_test_epoch22_csin.json"
    dest.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("convlstm", {k: metrics["csi"][k] for k in ("20", "30", "40")}, metrics["csi_dbz_hour"]["40"], flush=True)
    print("wrote", dest, flush=True)


def run_nowcast() -> None:
    root = Path("/root/autodl-fs/baselines/nowcastnet")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "capsule" / "code"))
    import yaml
    from eval_test import load_pair, make_loader
    from dataset import TEST_MONTHS

    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    data = cfg["data"]
    t_in, t_out = int(data["t_in"]), int(data["t_out"])
    ngf = int(cfg["model"]["ngf"])
    base_c = int(cfg["model"]["base_c"])
    hw = (int(data["h_size"]), int(data["w_size"]))
    noise_hw = (hw[0] // int(data["noise_scale"]), hw[1] // int(data["noise_scale"]))
    device = torch.device("cuda")
    test_ds, loader = make_loader(data["data_root"], TEST_MONTHS, t_in, t_out, hw, 4)
    evo_path = root / "work_dirs/nowcastnet/evolution/last.pt"
    gen_path = root / "work_dirs/nowcastnet/generation/epoch_0001.pt"
    evo, generator, epoch = load_pair(evo_path, gen_path, device, t_in, t_out, base_c, ngf)
    per_batch, per_lead = fresh()
    print(f"nowcastnet samples={len(test_ds)} batches={len(loader)} epoch={epoch}", flush=True)
    with torch.no_grad():
        for i, frames in enumerate(loader):
            frames = frames.to(device, non_blocking=True)
            inputs = frames[:, :t_in]
            labels = frames[:, t_in:]
            from models import rollout
            evo_result = rollout(evo, inputs, t_in, t_out) / 128.0
            torch.manual_seed(1000 + i)
            noise = torch.randn(inputs.shape[0], ngf, noise_hw[0], noise_hw[1], device=device)
            pred = generator(inputs, evo_result, noise).clamp(0.0, 128.0).detach().cpu().numpy()
            true_r = labels.detach().cpu().numpy()
            accumulate(pred, true_r, per_batch, per_lead)
            if (i + 1) % 100 == 0:
                print(f"  nowcastnet {i + 1}/{len(loader)}", flush=True)
    metrics = summarize(per_batch, per_lead, len(test_ds))
    metrics.update({"model": "nowcastnet", "ckpt": gen_path.name, "epoch": epoch, "evolution_ckpt": "evolution/last.pt"})
    dest = root / "work_dirs/nowcastnet/eval_metrics_test_epoch01_csin.json"
    dest.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("nowcastnet", {k: metrics["csi"][k] for k in ("20", "30", "40")}, metrics["csi_dbz_hour"]["40"], flush=True)
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    which = sys.argv[1]
    {"ring16": run_ring16, "convlstm": run_convlstm, "nowcastnet": run_nowcast}[which]()
