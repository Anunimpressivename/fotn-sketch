"""Test-set metrics for g025 ring-softmax τ=16, best-val-CSI40 (epoch 4)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sketch.data import make_loader  # noqa: E402
from sketch.model import SketchModel  # noqa: E402
from sketch.normalize import NormStats  # noqa: E402


def csi_pod_far(pred: np.ndarray, gt: np.ndarray, thr: float):
    pb = pred >= thr
    tb = gt >= thr
    hits = int(np.logical_and(pb, tb).sum())
    miss = int(np.logical_and(tb, ~pb).sum())
    fa = int(np.logical_and(pb, ~tb).sum())
    denom = hits + miss + fa
    csi = hits / denom if denom else float("nan")
    pod = hits / (hits + miss) if (hits + miss) else float("nan")
    far = fa / (hits + fa) if (hits + fa) else float("nan")
    return csi, pod, far


def build(cfg: dict, ckpt_path: Path, norm: NormStats, device: torch.device) -> SketchModel:
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    model = SketchModel(
        t_in=dcfg["t_in"],
        t_out=dcfg["t_out"],
        base_ch=mcfg["base_ch"],
        flow_base_ch=mcfg.get("flow_base_ch"),
        a_init=mcfg["a_init"],
        u_scale=mcfg["u_scale"],
        delta_scale=mcfg["delta_scale"],
        clamp_state=mcfg["clamp_state"],
        clamp_s=mcfg["clamp_s"],
        use_precip=bool(dcfg.get("use_precip", False)),
        residual_cond=bool(mcfg.get("residual_cond", False)),
        detach_phys_cond=bool(mcfg.get("detach_phys_cond", False)),
        res_base_ch=mcfg.get("res_base_ch"),
        n_noise=int(mcfg.get("n_noise", 4)),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.radar_mean.fill_(float(norm.radar_mean))
    model.radar_std.fill_(float(norm.radar_std))
    model.eval()
    return model

RUN = Path("/root/autodl-fs/fotn-sketch1/runs/ring16")
CFG = ROOT / "configs/config_rp_flow32_resgan_gdisc_g025_ring16.yaml"
CKPT = RUN / "best_csi40.pt"
SCALE = 128.0
TH = (8.0, 51.2, 76.8, 102.4)


def evaluate(model, loader, device: torch.device) -> dict:
    model.eval()
    leads = loader.dataset.t_out
    acc_mae = []
    acc = {t: {"csi": [], "pod": [], "far": []} for t in TH}
    csi_t = {t: [[] for _ in range(leads)] for t in TH}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="test", leave=False)):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            torch.manual_seed(1000 + i)
            _fields, traj = model.forward_batch(batch, "gan")
            pred = (traj["radar"].clamp(0, 1) * SCALE).cpu().numpy()
            gt = (batch["radar_gt"].clamp(0, 1) * SCALE).cpu().numpy()
            acc_mae.append(float(np.abs(pred - gt).mean()))
            for t in TH:
                c, po, fa = csi_pod_far(pred, gt, t)
                acc[t]["csi"].append(c)
                acc[t]["pod"].append(po)
                acc[t]["far"].append(fa)
            for li in range(leads):
                for t in TH:
                    c, _, _ = csi_pod_far(pred[:, li], gt[:, li], t)
                    csi_t[t][li].append(c)

    def nanmean(xs):
        a = np.array(xs, dtype=np.float64)
        return float(np.nanmean(a)) if a.size else float("nan")

    out = {
        "num_samples": len(loader.dataset),
        "mae": float(np.mean(acc_mae)),
        "csi": {str(t): nanmean(acc[t]["csi"]) for t in TH},
        "pod": {str(t): nanmean(acc[t]["pod"]) for t in TH},
        "far": {str(t): nanmean(acc[t]["far"]) for t in TH},
        "csi_dbz_hour": {},
        "csi_by_lead": {},
        "lead_minutes": [6 * (i + 1) for i in range(leads)],
    }
    for t, name in ((8.0, "8"), (51.2, "20"), (76.8, "30"), (102.4, "40")):
        series = [nanmean(csi_t[t][i]) for i in range(leads)]
        out["csi_by_lead"][name] = series
        out["csi_dbz_hour"][name] = {
            "1h": float(np.mean(series[:10])),
            "2h": float(np.mean(series[:20])),
            "3h": float(np.mean(series)),
        }
    return out


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    dcfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = NormStats.load(RUN / "norm_stats.json")
    loader = make_loader(
        dcfg["eval_root"], dcfg["test_months"], norm,
        dcfg["t_in"], dcfg["t_out"], tuple(dcfg["image_hw"]),
        dcfg["batch_size"], dcfg["num_workers"], False, use_precip=False,
    )
    print(f"device={device} samples={len(loader.dataset)} batches={len(loader)} ckpt={CKPT.name}", flush=True)
    model = build(cfg, CKPT, norm, device)
    metrics = evaluate(model, loader, device)
    metrics["ckpt"] = CKPT.name
    metrics["epoch"] = 4
    dest = RUN / "eval_metrics_test_epoch04.json"
    dest.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    h20, h30, h40 = metrics["csi_dbz_hour"]["20"], metrics["csi_dbz_hour"]["30"], metrics["csi_dbz_hour"]["40"]
    print(
        f"ep4 MAE={metrics['mae']:.3f} "
        f"CSI8={metrics['csi']['8.0']:.3f} "
        f"CSI20={metrics['csi']['51.2']:.3f} CSI30={metrics['csi']['76.8']:.3f} CSI40={metrics['csi']['102.4']:.3f}",
        flush=True,
    )
    print(
        f"  20dBZ {h20['1h']:.3f}/{h20['2h']:.3f}/{h20['3h']:.3f}  "
        f"30dBZ {h30['1h']:.3f}/{h30['2h']:.3f}/{h30['3h']:.3f}  "
        f"40dBZ {h40['1h']:.3f}/{h40['2h']:.3f}/{h40['3h']:.3f}",
        flush=True,
    )
    print(
        f"  POD/FAR 20 {metrics['pod']['51.2']:.3f}/{metrics['far']['51.2']:.3f}  "
        f"30 {metrics['pod']['76.8']:.3f}/{metrics['far']['76.8']:.3f}  "
        f"40 {metrics['pod']['102.4']:.3f}/{metrics['far']['102.4']:.3f}",
        flush=True,
    )
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
