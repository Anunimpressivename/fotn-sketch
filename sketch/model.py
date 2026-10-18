from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics import rollout
from .unet import SmallUNet


class SketchModel(nn.Module):
    def __init__(
        self,
        t_in: int = 10,
        t_out: int = 30,
        base_ch: int = 16,
        flow_base_ch: int | None = None,
        a_init: float = 1.0,
        u_scale: float = 2.0,
        delta_scale: float = 0.25,
        clamp_state: float = 1.5,
        clamp_s: float = 1.5,
        use_precip: bool = True,
        residual_cond: bool = False,
        detach_phys_cond: bool = False,
        res_base_ch: int | None = None,
        n_noise: int = 4,
    ):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.u_scale = u_scale
        self.delta_scale = delta_scale
        self.clamp_state = clamp_state
        self.clamp_s = clamp_s
        self.use_precip = use_precip
        self.residual_cond = bool(residual_cond)
        self.detach_phys_cond = bool(detach_phys_cond)
        self.n_noise = int(n_noise)
        conv_in = (3 * t_in) if use_precip else (2 * t_in)
        flow_ch = base_ch if flow_base_ch is None else int(flow_base_ch)
        self.flow_base_ch = flow_ch
        res_ch = base_ch if res_base_ch is None else int(res_base_ch)
        self.res_base_ch = res_ch
        self.flow = SmallUNet(2 * t_in, t_out * 4, flow_ch)
        self.conv = SmallUNet(conv_in, t_out * 2, base_ch)
        res_in = 2 * t_in
        if self.residual_cond:
            res_in = 2 * t_in + t_out + self.n_noise
        self.res = SmallUNet(res_in, t_out * 2, res_ch)
        if self.residual_cond and self.n_noise > 0:
            # Noise channels start at zero so the supervised residual is not jittered.
            nn.init.zeros_(self.res.enc0.net[0].weight[:, -self.n_noise :])
        a0 = torch.tensor(float(a_init)).clamp_min(1e-4)
        self.a_raw = nn.Parameter(torch.log(torch.expm1(a0)))
        self.register_buffer("radar_mean", torch.tensor(0.0))
        self.register_buffer("radar_std", torch.tensor(1.0))

    @property
    def a(self) -> torch.Tensor:
        return F.softplus(self.a_raw)

    def _time_maps(self, x: torch.Tensor, ch_per_t: int) -> torch.Tensor:
        b, _, h, w = x.shape
        return x.view(b, self.t_out, ch_per_t, h, w)

    def encode(
        self,
        radar_z: torch.Tensor,
        pwv_z: torch.Tensor,
        precip_z: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        flow_in = torch.cat([radar_z, pwv_z], dim=1)
        if self.use_precip:
            if precip_z is None:
                raise ValueError("precip_z required when use_precip=True")
            conv_in = torch.cat([radar_z, pwv_z, precip_z], dim=1)
        else:
            conv_in = flow_in
        u = torch.tanh(self._time_maps(self.flow(flow_in), 4)) * self.u_scale
        ke = self._time_maps(self.conv(conv_in), 2)
        b, _, h, w = radar_z.shape
        if self.residual_cond:
            delta_radar = radar_z.new_zeros(b, self.t_out, h, w)
            delta_pwv = radar_z.new_zeros(b, self.t_out, h, w)
        else:
            dlt = torch.tanh(self._time_maps(self.res(flow_in), 2)) * self.delta_scale
            delta_radar = dlt[:, :, 0]
            delta_pwv = dlt[:, :, 1]
        return {
            "u_radar": u[:, :, 0:2],
            "u_pwv": u[:, :, 2:4],
            "K": ke[:, :, 0],
            "eta": ke[:, :, 1],
            "delta_radar": delta_radar,
            "delta_pwv": delta_pwv,
        }

    def _residual(
        self,
        radar_z: torch.Tensor,
        pwv_z: torch.Tensor,
        evo_radar: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = self.radar_std.clamp_min(1e-6)
        evo_z = (evo_radar - self.radar_mean) / std
        x = torch.cat([radar_z, pwv_z, evo_z, noise], dim=1)
        dlt = torch.tanh(self._time_maps(self.res(x), 2)) * self.delta_scale
        return dlt[:, :, 0], dlt[:, :, 1]

    def configure(self, stage: str) -> None:
        train_flow = True
        train_conv = stage in ("conv", "res", "gan")
        train_res = stage in ("res", "gan")
        train_a = stage in ("conv", "res")
        for p in self.flow.parameters():
            p.requires_grad = train_flow
        for p in self.conv.parameters():
            p.requires_grad = train_conv
        for p in self.res.parameters():
            p.requires_grad = train_res
        self.a_raw.requires_grad = train_a

    def forward(
        self,
        radar_z: torch.Tensor,
        pwv_z: torch.Tensor,
        radar0: torch.Tensor,
        pwv0: torch.Tensor,
        stage: str,
        precip_z: torch.Tensor | None = None,
    ):
        fields = self.encode(radar_z, pwv_z, precip_z)
        if stage == "flow":
            fields["K"] = torch.full_like(fields["K"], -8.0)
            fields["eta"] = torch.full_like(fields["eta"], -8.0)
            fields["delta_radar"] = torch.zeros_like(fields["delta_radar"])
            fields["delta_pwv"] = torch.zeros_like(fields["delta_pwv"])
        elif stage == "conv":
            fields["delta_radar"] = torch.zeros_like(fields["delta_radar"])
            fields["delta_pwv"] = torch.zeros_like(fields["delta_pwv"])
        if self.residual_cond and stage in ("res", "gan"):
            traj0 = rollout(
                radar0, pwv0,
                fields["u_radar"], fields["u_pwv"],
                fields["K"], fields["eta"],
                fields["delta_radar"], fields["delta_pwv"],
                self.a, self.clamp_state, self.clamp_s,
            )
            if self.training or stage == "gan":
                noise = torch.randn(
                    radar_z.shape[0], self.n_noise, radar_z.shape[-2], radar_z.shape[-1],
                    device=radar_z.device, dtype=radar_z.dtype,
                )
            else:
                noise = torch.zeros(
                    radar_z.shape[0], self.n_noise, radar_z.shape[-2], radar_z.shape[-1],
                    device=radar_z.device, dtype=radar_z.dtype,
                )
            evo_for_res = traj0["radar"]
            if self.detach_phys_cond:
                evo_for_res = evo_for_res.detach()
            d_radar, d_pwv = self._residual(radar_z, pwv_z, evo_for_res, noise)
            fields["delta_radar"] = d_radar
            fields["delta_pwv"] = d_pwv
            fields["evo_radar"] = traj0["radar"]
        traj = rollout(
            radar0, pwv0,
            fields["u_radar"], fields["u_pwv"],
            fields["K"], fields["eta"],
            fields["delta_radar"], fields["delta_pwv"],
            self.a, self.clamp_state, self.clamp_s,
        )
        return fields, traj

    def forward_batch(self, batch: dict, stage: str):
        return self.forward(
            batch["radar_in"],
            batch["pwv_in"],
            batch["radar_hist"][:, -1],
            batch["pwv_hist"][:, -1],
            stage,
            precip_z=batch.get("precip_in"),
        )
