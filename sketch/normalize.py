from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class NormStats:
    radar_mean: float = 0.0
    radar_std: float = 1.0
    pwv_mean: float = 0.0
    pwv_std: float = 1.0
    precip_mean: float = 0.0
    precip_std: float = 1.0

    def encode(self, x, modality: str):
        mean, std = self._ms(modality)
        return (x - mean) / max(std, 1e-6)

    def decode(self, x, modality: str):
        mean, std = self._ms(modality)
        return x * max(std, 1e-6) + mean

    def _ms(self, modality: str):
        if modality == "radar":
            return self.radar_mean, self.radar_std
        if modality == "pwv":
            return self.pwv_mean, self.pwv_std
        if modality == "precip":
            return self.precip_mean, self.precip_std
        raise KeyError(modality)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "NormStats":
        return NormStats(**json.loads(Path(path).read_text(encoding="utf-8")))
