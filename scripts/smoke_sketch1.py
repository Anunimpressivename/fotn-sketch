from __future__ import annotations

import torch

from sketch.losses import total_loss
from sketch.model import SketchModel


def _run(use_precip: bool) -> None:
    torch.manual_seed(0)
    b, tin, tout, h, w = 2, 10, 30, 64, 64
    model = SketchModel(t_in=tin, t_out=tout, base_ch=16, use_precip=use_precip)
    radar_z = torch.randn(b, tin, h, w)
    pwv_z = torch.randn(b, tin, h, w)
    precip_z = torch.randn(b, tin, h, w) if use_precip else None
    radar0 = torch.rand(b, h, w)
    pwv0 = torch.rand(b, h, w)
    model.configure("flow")
    fields, traj = model(radar_z, pwv_z, radar0, pwv0, "flow", precip_z=precip_z)
    assert traj["radar"].shape == (b, tout, h, w)
    assert traj["precip"].max().item() < 1e-3, traj["precip"].max()
    print("flow forward ok use_precip", use_precip, "precip max", float(traj["precip"].max()), "params", sum(p.numel() for p in model.parameters()))

    batch = {
        "radar_hist": torch.rand(b, tin, h, w),
        "pwv_hist": torch.rand(b, tin, h, w),
        "radar_gt": torch.rand(b, tout, h, w),
        "pwv_gt": torch.rand(b, tout, h, w),
        "radar_in": radar_z,
        "pwv_in": pwv_z,
    }
    cfg = {
        "loss": {
            "advect_radar": 1.0, "advect_pwv": 4.0, "smooth": 0.05,
            "evo_radar": 1.0, "evo_pwv": 1.0, "evo_precip": 0.4,
            "gate": 0.15, "gate_leads": 4,
        }
    }
    if use_precip:
        batch["precip_hist"] = torch.rand(b, tin, h, w)
        batch["precip_gt"] = torch.rand(b, tout, h, w)
        batch["precip_in"] = precip_z
    for stage in ("flow", "conv", "res"):
        model.configure(stage)
        fields, traj = model.forward_batch(batch, stage)
        loss, logs = total_loss(traj, fields, batch, stage, cfg)
        loss.backward()
        assert torch.isfinite(loss), (stage, loss)
        assert ("evo_p" in logs) == use_precip or stage == "flow"
        if stage != "flow":
            assert ("evo_p" in logs) == use_precip, logs
        model.zero_grad(set_to_none=True)
        print(stage, "backward ok", use_precip, logs)


def main() -> None:
    _run(True)
    _run(False)
    print("smoke ok")


if __name__ == "__main__":
    main()
