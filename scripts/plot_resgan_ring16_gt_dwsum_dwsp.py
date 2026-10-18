"""Same four 202508 cases: GT / ring16 e4 / dwsum e5 / dwsp e6."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sketch.data import make_loader  # noqa: E402
from sketch.model import SketchModel  # noqa: E402
from sketch.normalize import NormStats  # noqa: E402

CASES = (3511, 2680, 3399, 4370)
LEADS_MIN = (30, 60, 120, 180)
SCALE = 128.0
DBZ_SCALE = 50.0 / 128.0
RUNS = Path("/root/autodl-fs/fotn-sketch1/runs")
CFG = ROOT / "configs/config_rp_flow32_resgan_gdisc_g025_ring16_dwsp.yaml"
FONT = Path("/tmp/fonts/NotoSansSC-Regular.otf")


def lead_index(minutes: int) -> int:
    return minutes // 6 - 1


def to_dbz(intensity: np.ndarray) -> np.ndarray:
    return np.clip(intensity, 0.0, SCALE) * DBZ_SCALE


def csi(pred: np.ndarray, gt: np.ndarray, thr: float) -> float:
    pb = pred >= thr
    tb = gt >= thr
    hits = int(np.logical_and(pb, tb).sum())
    miss = int(np.logical_and(tb, ~pb).sum())
    fa = int(np.logical_and(pb, ~tb).sum())
    denom = hits + miss + fa
    return hits / denom if denom else float("nan")


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
    model.load_state_dict(state["model"], strict=True)
    model.radar_mean.fill_(float(norm.radar_mean))
    model.radar_std.fill_(float(norm.radar_std))
    model.eval()
    return model


def forecast(model: SketchModel, batch: dict, stage: str) -> np.ndarray:
    _fields, traj = model.forward_batch(batch, stage)
    return (traj["radar"].clamp(0, 1) * SCALE)[0].cpu().numpy()


def gan_forecast(model: SketchModel, batch: dict, idx: int, device: torch.device) -> np.ndarray:
    torch.manual_seed(1000 + idx)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1000 + idx)
    return forecast(model, batch, "gan")


def plot_case(path: Path, title: str, columns: list[tuple[str, np.ndarray]]) -> None:
    gt = columns[0][1]
    fig, axes = plt.subplots(len(LEADS_MIN), len(columns), figsize=(15.4, 12.6), constrained_layout=True)
    im = None
    for r, minutes in enumerate(LEADS_MIN):
        t = lead_index(minutes)
        for c, (name, arr) in enumerate(columns):
            ax = axes[r, c]
            im = ax.imshow(to_dbz(arr[t]), cmap="turbo", vmin=0.0, vmax=50.0, origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=12)
            if c == 0:
                ax.set_ylabel(f"+{minutes} min", fontsize=11)
                note = f"max {to_dbz(arr[t]).max():.0f} dBZ"
            else:
                mae = float(np.mean(np.abs(arr[t] - gt[t])))
                note = f"MAE {mae:.1f}\nCSI20 {csi(arr[t], gt[t], 51.2):.2f}  CSI40 {csi(arr[t], gt[t], 102.4):.2f}"
            ax.text(
                0.02, 0.98, note, transform=ax.transAxes, va="top", ha="left", fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, ec="none"),
            )
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.012, label="dBZ")
    fig.suptitle(title, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    if FONT.is_file():
        font_manager.fontManager.addfont(str(FONT))
        plt.rcParams["font.family"] = "Noto Sans SC"
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    dcfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = NormStats.load(RUNS / "ring16" / "norm_stats.json")
    loader = make_loader(
        dcfg["eval_root"], dcfg["test_months"], norm,
        dcfg["t_in"], dcfg["t_out"], tuple(dcfg["image_hw"]),
        1, 0, False, use_precip=False,
    )
    ds = loader.dataset
    ring16 = build(cfg, RUNS / "ring16" / "best_csi40.pt", norm, device)
    dwsum = build(cfg, RUNS / "ring16_dwsum" / "best_csi40.pt", norm, device)
    dwsp = build(cfg, RUNS / "ring16_dwsp" / "best_csi40.pt", norm, device)
    out_dir = RUNS / "ring16_dwsp" / "cases" / "gt_ring16_dwsum"
    manifest = []
    with torch.no_grad():
        for rank, idx in enumerate(CASES, 1):
            sample = ds[idx]
            batch = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v for k, v in sample.items()}
            gt = (batch["radar_gt"].clamp(0, 1) * SCALE)[0].cpu().numpy()
            pred_ring16 = gan_forecast(ring16, batch, idx, device)
            pred_dwsum = gan_forecast(dwsum, batch, idx, device)
            pred_dwsp = gan_forecast(dwsp, batch, idx, device)
            t0 = ds.windows[idx][ds.t_in]
            stamp = t0.strftime("%Y-%m-%d %H:%M")
            path = out_dir / f"case{rank}_{t0.strftime('%Y%m%d_%H%M')}.png"
            plot_case(
                path,
                f"个例 {rank}  {stamp}  idx {idx}  测试 202508   原图 / ring16 第4轮 / dwsum 第5轮 / dwsp 第6轮",
                [
                    ("原图", gt),
                    ("ring16", pred_ring16),
                    ("dwsum", pred_dwsum),
                    ("dwsp", pred_dwsp),
                ],
            )
            manifest.append({"rank": rank, "idx": idx, "start": stamp, "path": str(path)})
            print("wrote", path, flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("done", out_dir, flush=True)


if __name__ == "__main__":
    main()
