"""Test-set metrics for ring16_imax, best validation CSI@40 (GAN local epoch 11)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_resgan_g025_ring16_ep4_test import build, evaluate  # noqa: E402
from sketch.data import make_loader  # noqa: E402
from sketch.normalize import NormStats  # noqa: E402
import torch  # noqa: E402

RUN = Path("/root/autodl-fs/fotn-sketch1/runs/ring16_imax")
CFG = ROOT / "configs/config_rp_flow32_resgan_gdisc_g025_ring16_imax.yaml"
CKPT = RUN / "best_csi40.pt"
GAN_LOCAL_EPOCH = 11


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
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    global_epoch = int(state["epoch"])
    print(
        f"device={device} samples={len(loader.dataset)} batches={len(loader)} "
        f"ckpt={CKPT.name} global_epoch={global_epoch} gan_local={GAN_LOCAL_EPOCH} "
        f"val_csi40={float(state.get('val_csi40', float('nan'))):.6f}",
        flush=True,
    )
    model = build(cfg, CKPT, norm, device)
    metrics = evaluate(model, loader, device)
    metrics["ckpt"] = CKPT.name
    metrics["epoch"] = GAN_LOCAL_EPOCH
    metrics["global_epoch"] = global_epoch
    metrics["val_csi40"] = float(state["val_csi40"])
    dest = RUN / f"eval_metrics_test_epoch{GAN_LOCAL_EPOCH:02d}.json"
    dest.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    h20 = metrics["csi_dbz_hour"]["20"]
    h30 = metrics["csi_dbz_hour"]["30"]
    h40 = metrics["csi_dbz_hour"]["40"]
    print(
        f"ep{GAN_LOCAL_EPOCH} MAE={metrics['mae']:.3f} "
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
