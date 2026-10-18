from __future__ import annotations

import re
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .normalize import NormStats

TS_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})")
STEP = timedelta(minutes=6)


def discover_roots(data_root: str | Path, use_precip: bool = True) -> Dict[str, Path]:
    root = Path(data_root)
    found: Dict[str, Path] = {}
    for p in root.iterdir():
        if not p.is_dir():
            continue
        u = p.name.upper()
        if "PWV" in u:
            found["pwv"] = p
        elif "RADAR" in u or "雷达" in p.name:
            found["radar"] = p
        elif "RAIN" in u or "PRECIP" in u or "降水" in p.name:
            found["precip"] = p
    need = {"radar", "pwv", "precip"} if use_precip else {"radar", "pwv"}
    missing = need - set(found)
    if missing:
        raise FileNotFoundError(f"missing {missing} under {root}")
    return found


def _parse_ts(name: str) -> Optional[datetime]:
    m = TS_RE.search(name)
    if not m:
        return None
    y, mo, d, h, mi, s = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s)


def index_month(mod_root: Path, months: Sequence[str]) -> Dict[datetime, Path]:
    want = set(months)
    out: Dict[datetime, Path] = {}
    for month_dir in sorted(mod_root.iterdir()):
        if not month_dir.is_dir() or month_dir.name not in want:
            continue
        for fp in month_dir.rglob("*.png"):
            ts = _parse_ts(fp.name)
            if ts is not None:
                out[ts] = fp
    return out


@lru_cache(maxsize=8192)
def _load_intensity(path: str, th: int, tw: int) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    inten = (255.0 - arr) / 255.0
    h, w = inten.shape
    top = max((h - th) // 2, 0)
    left = max((w - tw) // 2, 0)
    crop = inten[top : top + th, left : left + tw]
    if crop.shape != (th, tw):
        pad = np.zeros((th, tw), dtype=np.float32)
        pad[: crop.shape[0], : crop.shape[1]] = crop
        crop = pad
    return crop


def load_intensity(path: Path, hw: Tuple[int, int]) -> np.ndarray:
    return _load_intensity(str(path), hw[0], hw[1]).copy()


def _precip_path(index: Dict[datetime, Path], ts: datetime) -> Optional[Path]:
    if ts in index:
        return index[ts]
    return index.get(ts.replace(minute=0, second=0, microsecond=0))


class WindowDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        months: Sequence[str],
        t_in: int = 10,
        t_out: int = 30,
        image_hw: Tuple[int, int] = (64, 64),
        norm: Optional[NormStats] = None,
        use_precip: bool = True,
    ) -> None:
        self.use_precip = use_precip
        self.roots = discover_roots(data_root, use_precip=use_precip)
        self.t_in = t_in
        self.t_out = t_out
        self.hw = image_hw
        self.norm = norm or NormStats()
        self.radar = index_month(self.roots["radar"], months)
        self.pwv = index_month(self.roots["pwv"], months)
        self.precip = index_month(self.roots["precip"], months) if use_precip else {}
        self.windows = self._windows()

    def _windows(self) -> List[List[datetime]]:
        need = self.t_in + self.t_out
        common = sorted(set(self.radar) & set(self.pwv))
        have = set(common)
        out: List[List[datetime]] = []
        for t0 in common:
            stamps = [t0 + STEP * k for k in range(need)]
            if any(t not in have for t in stamps):
                continue
            if self.use_precip and any(_precip_path(self.precip, t) is None for t in stamps):
                continue
            out.append(stamps)
        return out

    def __len__(self) -> int:
        return len(self.windows)

    def _stack(self, index: Dict[datetime, Path], stamps: List[datetime], precip: bool) -> torch.Tensor:
        frames = []
        for ts in stamps:
            p = _precip_path(index, ts) if precip else index[ts]
            frames.append(load_intensity(p, self.hw))
        return torch.from_numpy(np.stack(frames, axis=0))

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        stamps = self.windows[i]
        tin = self.t_in
        radar = self._stack(self.radar, stamps, False)
        pwv = self._stack(self.pwv, stamps, False)
        item = {
            "radar_in": self.norm.encode(radar[:tin], "radar"),
            "pwv_in": self.norm.encode(pwv[:tin], "pwv"),
            "radar_gt": radar[tin:],
            "pwv_gt": pwv[tin:],
            "radar_hist": radar[:tin],
            "pwv_hist": pwv[:tin],
        }
        if self.use_precip:
            precip = self._stack(self.precip, stamps, True)
            item["precip_in"] = self.norm.encode(precip[:tin], "precip")
            item["precip_gt"] = precip[tin:]
            item["precip_hist"] = precip[:tin]
        return item


def gather_norm(ds: WindowDataset, max_windows: int = 256) -> NormStats:
    n = min(len(ds), max_windows)
    step = max(len(ds) // n, 1)
    keys = ("radar", "pwv", "precip") if ds.use_precip else ("radar", "pwv")
    acc = {k: [] for k in keys}
    for i in range(0, len(ds), step):
        if len(acc["radar"]) >= n:
            break
        w = ds.windows[i]
        tin = ds.t_in
        acc["radar"].append(ds._stack(ds.radar, w[:tin], False).numpy())
        acc["pwv"].append(ds._stack(ds.pwv, w[:tin], False).numpy())
        if ds.use_precip:
            acc["precip"].append(ds._stack(ds.precip, w[:tin], True).numpy())
    stats = {}
    for k, chunks in acc.items():
        x = np.concatenate([c.reshape(-1) for c in chunks])
        stats[f"{k}_mean"] = float(x.mean())
        stats[f"{k}_std"] = float(max(x.std(), 1e-3))
    return NormStats(**stats)


def make_loader(
    data_root: str | Path,
    months: Sequence[str],
    norm: NormStats,
    t_in: int,
    t_out: int,
    image_hw: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    use_precip: bool = True,
) -> DataLoader:
    ds = WindowDataset(
        data_root, months, t_in, t_out, tuple(image_hw), norm, use_precip=use_precip,
    )
    kw = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if num_workers > 0:
        kw["prefetch_factor"] = 2
    return DataLoader(**kw)
