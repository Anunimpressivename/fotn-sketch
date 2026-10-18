from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .patch_loss import change_pixel_weight
from .physics import advect, divergence


def _l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().mean()


def _radar_change_weight(batch: dict, cfg: dict) -> torch.Tensor | None:
    """Radar pixel weight, or None when this run keeps uniform radar losses."""
    la = cfg.get("loss", {})
    if not bool(la.get("radar_change_weight", False)):
        return None
    return change_pixel_weight(
        batch["radar_gt"],
        batch["radar_hist"][:, -1],
        floor=float(la.get("change_weight_floor", 1.0)),
        w_max=float(la.get("change_weight_max", 64.0)),
        scale=float(la.get("change_weight_scale", 128.0)),
        mode=str(la.get("change_weight_mode", "abs")),
        delta_bias=float(la.get("change_weight_delta_bias", 16.0)),
        denom=float(la.get("change_weight_denom", 64.0)),
        spatial_kernel=int(la.get("change_weight_spatial", 1)),
        temporal_radius=int(la.get("change_weight_temporal", 0)),
        normalize_mean=bool(la.get("change_weight_normalize", False)),
    )


def advect_loss(
    hist: torch.Tensor,
    gt: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    cur = torch.cat([hist[:, -1:], gt[:, :-1]], dim=1)
    t_out = u.shape[1]
    acc = cur.new_zeros(())
    for t in range(t_out):
        pred = advect(cur[:, t : t + 1], u[:, t])
        err = (pred - gt[:, t : t + 1]).abs()
        if weight is None:
            acc = acc + err.mean()
        else:
            acc = acc + (err * weight[:, t : t + 1]).mean()
    return acc / t_out


def evo_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    lowpass_k: int = 1,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    k = max(int(lowpass_k), 1)
    if k <= 1:
        if weight is None:
            return _l1(pred, gt)
        return ((pred - gt).abs() * weight).mean()
    pad = k // 2
    pred_l = F.avg_pool2d(pred, kernel_size=k, stride=1, padding=pad)
    gt_l = F.avg_pool2d(gt, kernel_size=k, stride=1, padding=pad)
    if pred_l.shape[-2:] != pred.shape[-2:]:
        pred_l = F.interpolate(pred_l, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        gt_l = F.interpolate(gt_l, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    if weight is None:
        return _l1(pred_l, gt_l)
    w = F.avg_pool2d(weight, kernel_size=k, stride=1, padding=pad)
    if w.shape[-2:] != pred_l.shape[-2:]:
        w = F.interpolate(w, size=pred_l.shape[-2:], mode="bilinear", align_corners=False)
    return ((pred_l - gt_l).abs() * w).mean()


def residual_lowpass_l1(
    delta_radar: torch.Tensor,
    delta_pwv: torch.Tensor,
    kernel: int = 5,
    highpass_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Spatial low-pass L1 on residual deltas. high-pass optional.

    Returns (loss, low_l1, high_l1). Deltas are (B, T, H, W).
    """
    k = max(int(kernel), 1)
    hp_w = float(highpass_weight)

    def split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if k <= 1:
            return x, x.new_zeros(())
        pad = k // 2
        low = F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
        if low.shape[-2:] != x.shape[-2:]:
            low = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return low, x - low

    terms = []
    low_acc = []
    high_acc = []
    for x in (delta_radar, delta_pwv):
        low, high = split(x)
        low_m = low.abs().mean()
        high_m = high.abs().mean() if torch.is_tensor(high) and high.ndim else x.new_zeros(())
        low_acc.append(low_m)
        high_acc.append(high_m)
        terms.append(low_m + hp_w * high_m)
    loss = terms[0] + terms[1]
    return loss, low_acc[0] + low_acc[1], high_acc[0] + high_acc[1]


def smooth_loss(u_radar: torch.Tensor, u_pwv: torch.Tensor) -> torch.Tensor:
    u = torch.cat([u_radar, u_pwv], dim=2)
    dx = (u[:, :, :, :, 1:] - u[:, :, :, :, :-1]).abs().mean()
    dy = (u[:, :, :, 1:, :] - u[:, :, :, :-1, :]).abs().mean()
    dt = (u[:, 1:] - u[:, :-1]).abs().mean()
    return dx + dy + dt


def gate_loss(
    g: torch.Tensor,
    pwv_hist: torch.Tensor,
    pwv_gt: torch.Tensor,
    u_pwv: torch.Tensor,
    n_leads: int,
) -> torch.Tensor:
    u = u_pwv.detach()
    cur = torch.cat([pwv_hist[:, -1:], pwv_gt[:, :-1]], dim=1).detach()
    nxt = pwv_gt.detach()
    t_use = min(n_leads, u.shape[1])
    acc = g.new_zeros(())
    wsum = 0.0
    for t in range(t_use):
        conv = torch.relu(-divergence(u[:, t]) * cur[:, t : t + 1])
        sink = advect(cur[:, t : t + 1], u[:, t]) - nxt[:, t : t + 1]
        mask = (conv > 0).to(dtype=g.dtype)
        star = (sink > 0).to(dtype=g.dtype)
        if mask.sum() < 1:
            continue
        pred = g[:, t : t + 1]
        acc = acc + ((pred - star).abs() * mask).sum() / mask.sum().clamp_min(1.0)
        wsum += 1.0
    if wsum == 0:
        return g.new_zeros(())
    return acc / wsum


def total_loss(traj, fields, batch, stage: str, cfg: dict) -> tuple[torch.Tensor, Dict[str, float]]:
    la = cfg["loss"]
    w_radar = _radar_change_weight(batch, cfg)
    adv_r = advect_loss(batch["radar_hist"], batch["radar_gt"], fields["u_radar"], weight=w_radar)
    adv_v = advect_loss(batch["pwv_hist"], batch["pwv_gt"], fields["u_pwv"])
    sm = smooth_loss(fields["u_radar"], fields["u_pwv"])
    parts = {
        "adv_r": adv_r,
        "adv_v": adv_v,
        "smooth": sm,
    }
    loss = la["advect_radar"] * adv_r + la["advect_pwv"] * adv_v + la["smooth"] * sm
    if stage in ("conv", "res", "gan"):
        evo_k = int(la.get("evo_lowpass_k", 1))
        er = evo_loss(traj["radar"], batch["radar_gt"], lowpass_k=evo_k, weight=w_radar)
        ev = evo_loss(traj["pwv"], batch["pwv_gt"], lowpass_k=evo_k)
        gt = gate_loss(traj["g"], batch["pwv_hist"], batch["pwv_gt"], fields["u_pwv"], int(la["gate_leads"]))
        parts.update({"evo_r": er, "evo_v": ev, "gate": gt})
        evo_radar_w = float(la["evo_radar"])
        if stage == "gan" and bool(la.get("gan_drop_evo_radar", False)):
            evo_radar_w = 0.0
        loss = loss + evo_radar_w * er + la["evo_pwv"] * ev + la["gate"] * gt
        if "precip_gt" in batch:
            ep = evo_loss(traj["precip"], batch["precip_gt"])
            parts["evo_p"] = ep
            loss = loss + float(la.get("evo_precip", 0.0)) * ep
    logs = {k: float(v.detach()) for k, v in parts.items()}
    logs["loss"] = float(loss.detach())
    return loss, logs


def select_metric(traj, batch, cfg: dict | None = None) -> float:
    w_radar = None if cfg is None else _radar_change_weight(batch, cfg)
    radar = evo_loss(traj["radar"], batch["radar_gt"], weight=w_radar)
    pwv = evo_loss(traj["pwv"], batch["pwv_gt"])
    return float((radar + pwv).detach())
