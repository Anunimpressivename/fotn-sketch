from __future__ import annotations

import torch
import torch.nn.functional as F


def _grid(h: int, w: int, device, dtype):
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    return ys, xs


def advect(field: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """field (B,1,H,W), u (B,2,H,W) in grid-cells / step. Sample at x-u."""
    b, _, h, w = field.shape
    ys, xs = _grid(h, w, field.device, field.dtype)
    dx = u[:, 0] * (2.0 / max(w - 1, 1))
    dy = u[:, 1] * (2.0 / max(h - 1, 1))
    gx = xs.unsqueeze(0).expand(b, -1, -1) - dx
    gy = ys.unsqueeze(0).expand(b, -1, -1) - dy
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)


def divergence(u: torch.Tensor) -> torch.Tensor:
    """u (B,2,H,W) -> (B,1,H,W) central differences."""
    ux = u[:, 0:1]
    uy = u[:, 1:2]
    ux_p = F.pad(ux, (1, 1, 0, 0), mode="replicate")
    uy_p = F.pad(uy, (0, 0, 1, 1), mode="replicate")
    dux = 0.5 * (ux_p[:, :, :, 2:] - ux_p[:, :, :, :-2])
    duy = 0.5 * (uy_p[:, :, 2:, :] - uy_p[:, :, :-2, :])
    return dux + duy


def step(
    radar: torch.Tensor,
    pwv: torch.Tensor,
    u_radar: torch.Tensor,
    u_pwv: torch.Tensor,
    k: torch.Tensor,
    eta: torch.Tensor,
    d_radar: torch.Tensor,
    d_pwv: torch.Tensor,
    a: torch.Tensor,
    clamp_state: float,
    clamp_s: float,
):
    if radar.dim() == 3:
        radar = radar.unsqueeze(1)
        pwv = pwv.unsqueeze(1)
    if k.dim() == 3:
        k = k.unsqueeze(1)
        eta = eta.unsqueeze(1)
        d_radar = d_radar.unsqueeze(1)
        d_pwv = d_pwv.unsqueeze(1)

    g = torch.sigmoid(k)
    conv = torch.relu(-divergence(u_pwv) * pwv)
    s = (a.view(-1, 1, 1, 1) * conv * g).clamp(max=clamp_s)
    m = F.softplus(eta)
    keep = s / (1.0 + m)
    rain = s * m / (1.0 + m)

    radar = torch.relu(advect(radar, u_radar) + keep + d_radar).clamp(max=clamp_state)
    pwv = torch.relu(advect(pwv, u_pwv) - s + d_pwv).clamp(max=clamp_state)
    precip = torch.relu(rain).clamp(max=clamp_state)
    return radar.squeeze(1), pwv.squeeze(1), precip.squeeze(1), g.squeeze(1), conv.squeeze(1)


def rollout(
    radar0: torch.Tensor,
    pwv0: torch.Tensor,
    u_radar: torch.Tensor,
    u_pwv: torch.Tensor,
    k: torch.Tensor,
    eta: torch.Tensor,
    d_radar: torch.Tensor,
    d_pwv: torch.Tensor,
    a: torch.Tensor,
    clamp_state: float,
    clamp_s: float,
):
    t_out = u_radar.shape[1]
    r, v = radar0, pwv0
    rs, vs, ps, gs, cs = [], [], [], [], []
    for t in range(t_out):
        r, v, p, g, c = step(
            r, v, u_radar[:, t], u_pwv[:, t], k[:, t], eta[:, t],
            d_radar[:, t], d_pwv[:, t], a, clamp_state, clamp_s,
        )
        rs.append(r)
        vs.append(v)
        ps.append(p)
        gs.append(g)
        cs.append(c)
    return {
        "radar": torch.stack(rs, 1),
        "pwv": torch.stack(vs, 1),
        "precip": torch.stack(ps, 1),
        "g": torch.stack(gs, 1),
        "conv": torch.stack(cs, 1),
    }
