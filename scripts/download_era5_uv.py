"""Download ERA5 700/850 hPa wind for the ring16 test window."""
from __future__ import annotations

import cdsapi
from pathlib import Path

OUT = Path("/root/autodl-fs/fotn-sketch1/runs/ring16/era5")
OUT.mkdir(parents=True, exist_ok=True)

COMMON = {
    "product_type": ["reanalysis"],
    "variable": ["u_component_of_wind", "v_component_of_wind"],
    "pressure_level": ["700", "850"],
    "time": [f"{h:02d}:00" for h in range(24)],
    "area": [43.5, 112.5, 35.5, 120.5],
    "data_format": "netcdf",
    "download_format": "unarchived",
}

JOBS = [
    ("2025", "08", [f"{d:02d}" for d in range(1, 32)], OUT / "era5_uv_202508.nc"),
    ("2025", "07", ["31"], OUT / "era5_uv_20250731.nc"),
    ("2025", "09", ["01"], OUT / "era5_uv_20250901.nc"),
]


def main() -> None:
    client = cdsapi.Client()
    for year, month, days, target in JOBS:
        print(f"request {target.name}", flush=True)
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {**COMMON, "year": [year], "month": [month], "day": days},
            str(target),
        )
        print(f"wrote {target} size={target.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
