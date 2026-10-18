from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from sketch.data import make_loader
from sketch.model import SketchModel
from sketch.ncwrap import SketchFlowGen, build_generator
from sketch.normalize import NormStats

SCALE = 128.0
DBZ_MAX = 50.0


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    use_precip = bool(dcfg.get("use_precip", True))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    months = dcfg["val_months"] if args.split == "val" else dcfg["test_months"]
    root = dcfg["eval_root"]
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    norm = NormStats.load(Path(args.ckpt).resolve().parent / "norm_stats.json")
    loader = make_loader(
        root, months, norm, dcfg["t_in"], dcfg["t_out"], dcfg["image_hw"],
        dcfg["batch_size"], dcfg["num_workers"], False,
        use_precip=use_precip,
    )
    model = SketchModel(
        t_in=dcfg["t_in"], t_out=dcfg["t_out"], base_ch=mcfg["base_ch"],
        flow_base_ch=mcfg.get("flow_base_ch"),
        a_init=mcfg["a_init"], u_scale=mcfg["u_scale"], delta_scale=mcfg["delta_scale"],
        clamp_state=mcfg["clamp_state"], clamp_s=mcfg["clamp_s"],
        use_precip=use_precip,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    use_gen = ckpt.get("gen") is not None
    if use_gen:
        gen = build_generator(cfg).to(device)
        gen.load_state_dict(ckpt["gen"])
        model = SketchFlowGen(model, gen).to(device)
    model.eval()
    stage = "gen" if use_gen else str(ckpt.get("stage", "res"))

    th_disp = [8.0, 16.0, 32.0, 48.0, 64.0, 51.2, 76.8, 102.4]
    acc_mae = []
    acc = {t: {"csi": [], "pod": [], "far": []} for t in th_disp}
    leads = dcfg["t_out"]
    csi_t = {t: [[] for _ in range(leads)] for t in (51.2, 76.8, 102.4)}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval-{args.split}"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if stage == "gen":
                _, traj = model(batch)
            else:
                _, traj = model.forward_batch(batch, stage)
            pred = (traj["radar"].clamp(0, 1) * SCALE).cpu().numpy()
            gt = (batch["radar_gt"].clamp(0, 1) * SCALE).cpu().numpy()
            acc_mae.append(float(np.abs(pred - gt).mean()))
            for t in th_disp:
                c, po, fa = csi_pod_far(pred, gt, t)
                acc[t]["csi"].append(c)
                acc[t]["pod"].append(po)
                acc[t]["far"].append(fa)
            for li in range(leads):
                for t in (51.2, 76.8, 102.4):
                    c, _, _ = csi_pod_far(pred[:, li], gt[:, li], t)
                    csi_t[t][li].append(c)

    def nanmean(xs):
        a = np.array(xs, dtype=np.float64)
        return float(np.nanmean(a)) if a.size else float("nan")

    out = {
        "num_samples": len(loader.dataset),
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_stage": stage,
        "mae": float(np.mean(acc_mae)),
        "csi": {str(t): nanmean(acc[t]["csi"]) for t in th_disp},
        "pod": {str(t): nanmean(acc[t]["pod"]) for t in th_disp},
        "far": {str(t): nanmean(acc[t]["far"]) for t in th_disp},
        "csi_dbz_hour": {},
    }
    for t, name in ((51.2, "20"), (76.8, "30"), (102.4, "40")):
        series = [nanmean(csi_t[t][i]) for i in range(leads)]
        out["csi_dbz_hour"][name] = {
            "1h": float(np.mean(series[:10])),
            "2h": float(np.mean(series[:20])),
            "3h": float(np.mean(series)),
            "by_lead": series,
        }
    print(json.dumps({k: out[k] for k in ("mae", "csi", "pod", "far", "csi_dbz_hour") if k != "csi_dbz_hour"}, indent=2))
    print("dbz hour means", json.dumps(out["csi_dbz_hour"], indent=2)[:800])
    dest = Path(args.ckpt).resolve().parent / f"eval_metrics_{args.split}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
