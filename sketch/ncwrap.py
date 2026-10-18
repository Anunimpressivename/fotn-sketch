"""Frozen flow32 radar rollout + NowcastNet generator that replaces the radar forecast."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

from sketch.model import SketchModel

SCALE = 128.0
_FOTN_ROOT = Path("/root/autodl-fs/fotn")
_NC_CODE = Path("/root/autodl-fs/nowcastnet-hacks/code")


def _load(name: str, path: Path):
    if str(_NC_CODE) not in sys.path:
        sys.path.append(str(_NC_CODE))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_GEN = _load("sketch_nc_generator", _FOTN_ROOT / "fotn/models/nc_generator.py")
_DISC = _load("sketch_temporal_disc", _FOTN_ROOT / "fotn/models/temporal_disc.py")
NowcastNetGenerator = _GEN.NowcastNetGenerator
TemporalDiscriminator = _DISC.TemporalDiscriminator


def build_flow(cfg: dict, device: torch.device) -> SketchModel:
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    use_precip = bool(dcfg.get("use_precip", False))
    model = SketchModel(
        t_in=int(dcfg["t_in"]),
        t_out=int(dcfg["t_out"]),
        base_ch=int(mcfg["base_ch"]),
        flow_base_ch=mcfg.get("flow_base_ch"),
        a_init=float(mcfg["a_init"]),
        u_scale=float(mcfg["u_scale"]),
        delta_scale=float(mcfg["delta_scale"]),
        clamp_state=float(mcfg["clamp_state"]),
        clamp_s=float(mcfg["clamp_s"]),
        use_precip=use_precip,
    )
    return model.to(device)


def build_generator(cfg: dict) -> nn.Module:
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    hw = tuple(dcfg["image_hw"])
    return NowcastNetGenerator(
        t_in=int(dcfg["t_in"]),
        t_out=int(dcfg["t_out"]),
        ngf=int(mcfg.get("ngf", 32)),
        img_h=int(hw[0]),
        img_w=int(hw[1]),
    )


def build_discriminator(cfg: dict) -> nn.Module:
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    return TemporalDiscriminator(
        seq_len=int(dcfg["t_out"]),
        base_c=int(tcfg.get("gan_disc_base_c", 16)),
    )


class SketchFlowGen(nn.Module):
    """flow32 stays frozen. Generator emits the future radar field in [0, 128]."""

    def __init__(self, flow: SketchModel, gen: nn.Module):
        super().__init__()
        self.flow = flow
        self.gen = gen
        self.freeze_flow()

    def freeze_flow(self) -> None:
        self.flow.eval()
        for p in self.flow.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.flow.eval()
        return self

    @torch.no_grad()
    def evolve(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
        fields, traj = self.flow.forward_batch(batch, "res")
        hist = batch["radar_hist"].clamp(0.0, 1.0) * SCALE
        evo = traj["radar"].clamp(0.0, 1.0) * SCALE
        return hist, evo, fields, traj

    def generate(
        self,
        hist: torch.Tensor,
        evo: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.gen(hist, evo, noise)

    def forward(self, batch: dict, noise: torch.Tensor | None = None) -> tuple[dict, dict]:
        hist, evo, fields, traj = self.evolve(batch)
        pred = self.generate(hist, evo, noise).clamp(0.0, SCALE)
        out = dict(traj)
        out["radar_evo"] = traj["radar"]
        out["radar"] = pred / SCALE
        return fields, out
