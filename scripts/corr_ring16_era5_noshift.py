"""No-shift correlation: ring16 lead-0 flow vs ERA5 wind on the assumed grid.

u_radar is compared with 700 hPa, u_pwv with 850 hPa. Model y is southward
positive and is flipped before the comparison. Issue-time labels are paired
both as UTC and as Beijing time (UTC+8), because the PNG clock has no zone.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

FLOW = Path("/root/autodl-fs/fotn-sketch1/runs/ring16/flow_test_202508_2h/flows.npz")
ERA = Path("/root/autodl-fs/fotn-sketch1/runs/ring16/era5/era5_uv_202508.nc")
OUT = Path("/root/autodl-fs/fotn-sketch1/runs/ring16/flow_test_202508_2h/corr_noshift.json")

LON_W, LON_E = 113.0, 120.0
LAT_S, LAT_N = 36.0, 43.0
NATIVE_H, NATIVE_W = 66, 70
CROP = 64
STEP_S = 6 * 60
DBZ_MIN = 15.0
MIN_PIX = 50
MIN_SPEED = 2.0
M_PER_DEG_LAT = 111_132.0


def crop_centers() -> tuple[np.ndarray, np.ndarray]:
    dlon = (LON_E - LON_W) / NATIVE_W
    dlat = (LAT_N - LAT_S) / NATIVE_H
    top = (NATIVE_H - CROP) // 2
    left = (NATIVE_W - CROP) // 2
    cols = np.arange(CROP)
    rows = np.arange(CROP)
    lon = LON_W + (left + cols + 0.5) * dlon
    lat = LAT_N - (top + rows + 0.5) * dlat
    lon2, lat2 = np.meshgrid(lon, lat)
    return lat2.astype(np.float64), lon2.astype(np.float64)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < MIN_PIX:
        return np.nan
    if np.std(a) < 1e-3 or np.std(b) < 1e-3:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def angle_err_deg(ue, vn, ue_r, vn_r) -> float:
    sp_m = np.hypot(ue, vn)
    sp_e = np.hypot(ue_r, vn_r)
    keep = (sp_m >= MIN_SPEED) & (sp_e >= MIN_SPEED)
    if int(keep.sum()) < MIN_PIX:
        return np.nan
    ang_m = np.arctan2(vn[keep], ue[keep])
    ang_e = np.arctan2(vn_r[keep], ue_r[keep])
    d = np.abs((ang_m - ang_e + np.pi) % (2 * np.pi) - np.pi)
    return float(np.degrees(d).mean())


def summarize(rows: list[dict]) -> dict:
    def col(key):
        a = np.array([r[key] for r in rows], dtype=np.float64)
        ok = np.isfinite(a)
        if not ok.any():
            return {"n": 0, "mean": None, "median": None}
        v = a[ok]
        return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v))}

    return {k: col(k) for k in ("r_east", "r_north", "r_speed", "dir_err_deg", "n_pix", "spd_model", "spd_era")}


def main() -> None:
    flow = np.load(FLOW)
    issue = flow["issue_time"].astype("datetime64[s]")
    lat2, lon2 = crop_centers()
    m_lon = 111_320.0 * np.cos(np.deg2rad(lat2))
    dx = ((LON_E - LON_W) / NATIVE_W) * m_lon
    dy = ((LAT_N - LAT_S) / NATIVE_H) * M_PER_DEG_LAT
    pts = np.stack([lat2.ravel(), lon2.ravel()], axis=1)

    with h5py.File(ERA, "r") as f:
        lat = np.asarray(f["latitude"][:], dtype=np.float64)
        lon = np.asarray(f["longitude"][:], dtype=np.float64)
        order = np.argsort(lat)
        lat_a = lat[order]
        lev = np.asarray(f["pressure_level"][:], dtype=np.float64)
        lev_i = {int(round(v)): i for i, v in enumerate(lev)}
        times = np.asarray(f["valid_time"][:], dtype=np.int64)
        era_t = times.astype("datetime64[s]")
        t_index = {t: i for i, t in enumerate(era_t)}
        u = np.asarray(f["u"][:], dtype=np.float32)[:, :, order, :]
        v = np.asarray(f["v"][:], dtype=np.float32)[:, :, order, :]

    radar = flow["radar_issue"]
    pwv = flow["pwv_issue"]
    dbz = radar * 50.0

    def one_level(name: str, model: np.ndarray, level: int, mask_mode: str, shift_h: int) -> dict:
        li = lev_i[level]
        rows = []
        missing = 0
        delta = np.timedelta64(shift_h, "h")
        for i, t0 in enumerate(issue):
            key = t0 - delta
            ti = t_index.get(key)
            if ti is None:
                missing += 1
                continue
            iu = RegularGridInterpolator((lat_a, lon), u[ti, li], bounds_error=False, fill_value=np.nan)
            iv = RegularGridInterpolator((lat_a, lon), v[ti, li], bounds_error=False, fill_value=np.nan)
            ue_r = iu(pts).reshape(CROP, CROP)
            vn_r = iv(pts).reshape(CROP, CROP)
            ue = model[i, 0] * dx / STEP_S
            vn = -model[i, 1] * dy / STEP_S
            if mask_mode == "radar":
                mask = dbz[i] >= DBZ_MIN
            else:
                gy, gx = np.gradient(pwv[i])
                g = np.hypot(gx, gy)
                mask = g >= np.percentile(g, 75)
            ok = mask & np.isfinite(ue_r) & np.isfinite(vn_r)
            ue_m, vn_m = ue[ok], vn[ok]
            ue_e, vn_e = ue_r[ok], vn_r[ok]
            rows.append({
                "r_east": pearson(ue_m, ue_e),
                "r_north": pearson(vn_m, vn_e),
                "r_speed": pearson(np.hypot(ue_m, vn_m), np.hypot(ue_e, vn_e)),
                "dir_err_deg": angle_err_deg(ue_m, vn_m, ue_e, vn_e),
                "n_pix": int(ok.sum()),
                "spd_model": float(np.hypot(ue_m, vn_m).mean()) if ok.any() else np.nan,
                "spd_era": float(np.hypot(ue_e, vn_e).mean()) if ok.any() else np.nan,
            })
        return {
            "pair": name,
            "level_hpa": level,
            "clock_shift_hours": shift_h,
            "n_issue": int(issue.size),
            "n_matched": len(rows),
            "n_missing_era": missing,
            "summary": summarize(rows),
        }

    jobs = []
    for shift, zone in ((0, "utc"), (8, "cst")):
        jobs.append(one_level("u_radar", flow["u_radar"], 700, "radar", shift) | {"zone": zone})
        jobs.append(one_level("u_pwv", flow["u_pwv"], 850, "pwv_grad", shift) | {"zone": zone})

    payload = {
        "grid": "assumed native lon[113,120] lat[36,43], center crop, no extra shift",
        "dbz_min": DBZ_MIN,
        "min_pixels": MIN_PIX,
        "min_speed_m_s": MIN_SPEED,
        "results": jobs,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for job in jobs:
        s = job["summary"]
        print(
            f"{job['zone']:3} {job['pair']:8} {job['level_hpa']}hPa "
            f"matched {job['n_matched']} "
            f"rE {s['r_east']['mean']} rN {s['r_north']['mean']} "
            f"rS {s['r_speed']['mean']} dir {s['dir_err_deg']['mean']} "
            f"spd {s['spd_model']['mean']} vs {s['spd_era']['mean']}",
            flush=True,
        )
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
