"""Export ring16 lead-0 flow on the test set, one issue time every 2 hours.

Each sample is an independent forward from the previous 10 frames. The saved
fields are the displacement used to advect the last observed radar and PWV
over the next 6 minutes.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sketch.data import WindowDataset  # noqa: E402
from sketch.model import SketchModel  # noqa: E402
from sketch.normalize import NormStats  # noqa: E402

RUN = Path("/root/autodl-fs/fotn-sketch1/runs/ring16")
CFG = ROOT / "configs/config_rp_flow32_resgan_gdisc_g025_ring16.yaml"
CKPT = RUN / "best_csi40.pt"
OUT_DIR = RUN / "flow_test_202508_2h"
EVERY_HOURS = 2
BATCH = 16


def build(cfg: dict, norm: NormStats, device: torch.device) -> SketchModel:
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
    state = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.radar_mean.fill_(float(norm.radar_mean))
    model.radar_std.fill_(float(norm.radar_std))
    model.eval()
    return model


def select_issue_times(ds: WindowDataset, every_hours: int) -> list[int]:
    """Windows whose last input frame is on the clock, every `every_hours`."""
    picked = []
    for i, stamps in enumerate(ds.windows):
        issue = stamps[ds.t_in - 1]
        if issue.minute or issue.second:
            continue
        if issue.hour % every_hours:
            continue
        picked.append(i)
    return picked


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    dcfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = NormStats.load(RUN / "norm_stats.json")
    ds = WindowDataset(
        dcfg["eval_root"],
        dcfg["test_months"],
        dcfg["t_in"],
        dcfg["t_out"],
        tuple(dcfg["image_hw"]),
        norm,
        use_precip=False,
    )
    idxs = select_issue_times(ds, EVERY_HOURS)
    print(
        f"device={device} windows={len(ds)} kept={len(idxs)} ckpt={CKPT.name}",
        flush=True,
    )
    if not idxs:
        raise SystemExit("no issue times matched")

    model = build(cfg, norm, device)
    u_radar, u_pwv = [], []
    radar_issue, pwv_issue = [], []
    issue_time = []
    with torch.no_grad():
        for start in range(0, len(idxs), BATCH):
            batch_i = idxs[start : start + BATCH]
            radar_in, pwv_in = [], []
            for i in batch_i:
                item = ds[i]
                radar_in.append(item["radar_in"])
                pwv_in.append(item["pwv_in"])
                radar_issue.append(item["radar_hist"][-1].numpy())
                pwv_issue.append(item["pwv_hist"][-1].numpy())
                issue_time.append(ds.windows[i][ds.t_in - 1])
            fields = model.encode(
                torch.stack(radar_in).to(device),
                torch.stack(pwv_in).to(device),
            )
            u_radar.append(fields["u_radar"][:, 0].cpu().numpy())
            u_pwv.append(fields["u_pwv"][:, 0].cpu().numpy())
            print(f"  {min(start + BATCH, len(idxs))}/{len(idxs)}", flush=True)

    u_radar_a = np.concatenate(u_radar, axis=0).astype(np.float32)
    u_pwv_a = np.concatenate(u_pwv, axis=0).astype(np.float32)
    radar_a = np.stack(radar_issue, axis=0).astype(np.float32)
    pwv_a = np.stack(pwv_issue, axis=0).astype(np.float32)
    times = np.array(issue_time, dtype="datetime64[s]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "flows.npz",
        issue_time=times,
        u_radar=u_radar_a,
        u_pwv=u_pwv_a,
        radar_issue=radar_a,
        pwv_issue=pwv_a,
    )
    meta = {
        "ckpt": str(CKPT),
        "config": str(CFG),
        "split": "test",
        "months": list(dcfg["test_months"]),
        "n": int(u_radar_a.shape[0]),
        "every_hours": EVERY_HOURS,
        "lead_index": 0,
        "step_minutes": 6,
        "issue_time": "last input frame; this flow advects that frame across the next 6 minutes",
        "components": ["x_east", "y_south"],
        "units": "grid_cells_per_step",
        "u_scale": float(cfg["model"]["u_scale"]),
        "image_hw": list(dcfg["image_hw"]),
        "arrays": {
            "issue_time": list(times.shape),
            "u_radar": list(u_radar_a.shape),
            "u_pwv": list(u_pwv_a.shape),
            "radar_issue": list(radar_a.shape),
            "pwv_issue": list(pwv_a.shape),
        },
        "radar_issue": "intensity (255-pixel)/255 at issue time; display scale = intensity*128",
        "pwv_issue": "same intensity encoding as the training PNGs, at issue time",
        "written": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"wrote {OUT_DIR} n={meta['n']} "
        f"issue {times[0]} .. {times[-1]} "
        f"u_radar {u_radar_a.shape} max {np.abs(u_radar_a).max():.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
