"""Intensity-weighted 16x16 patch adversarial loss, plus max-pooled MSE."""
from __future__ import annotations

import torch
import torch.nn.functional as F

def _spread_temporal_max(weight: torch.Tensor, radius: int) -> torch.Tensor:
    """0.5 * (w + max over the current frame and +/- radius frames).

    radius <= 0 leaves w unchanged. The first and last frames only see the
    neighbor that exists. Zero padding does not lower the max when w >= 0.
    """
    r = int(radius)
    if r <= 0:
        return weight
    b, t, h, w = weight.shape
    src = weight.float().permute(0, 2, 3, 1).reshape(b * h * w, 1, t)
    pooled = F.max_pool1d(src, kernel_size=2 * r + 1, stride=1, padding=r)
    mixed = 0.5 * (src + pooled)
    return mixed.reshape(b, h, w, t).permute(0, 3, 1, 2).to(dtype=weight.dtype)


def _spread_spatial_max(weight: torch.Tensor, kernel: int) -> torch.Tensor:
    """0.5 * (w + MaxPool kxk). kernel <= 1 leaves w unchanged."""
    k = int(kernel)
    if k <= 1:
        return weight
    if k % 2 == 0:
        raise ValueError(f"change weight spatial kernel must be odd, got {k}")
    b, t, h, w = weight.shape
    src = weight.float().reshape(b * t, 1, h, w)
    pooled = F.max_pool2d(src, kernel_size=k, stride=1, padding=k // 2)
    mixed = 0.5 * (src + pooled)
    return mixed.reshape(b, t, h, w).to(dtype=weight.dtype)


def _div_mean(weight: torch.Tensor) -> torch.Tensor:
    src = weight.float()
    return (src / src.mean().clamp_min(1e-6)).to(dtype=weight.dtype)


def change_pixel_weight(
    real: torch.Tensor,
    hist_last: torch.Tensor,
    *,
    floor: float = 1.0,
    w_max: float = 64.0,
    scale: float = 1.0,
    mode: str = "abs",
    delta_bias: float = 16.0,
    denom: float = 64.0,
    spatial_kernel: int = 1,
    temporal_radius: int = 0,
    normalize_mean: bool = False,
) -> torch.Tensor:
    """Detached radar pixel weight. Lead 0 uses hist_last; later leads use the previous GT frame.

    real: (B, T, H, W). hist_last: (B, H, W) or (B, 1, H, W).
    `scale` maps stored [0, 1] intensity onto the [0, 128] display scale.
    Pass scale=1 when `real` is already on that scale.

    mode "abs": min(w_max, floor + |I_t - I_prev|).
    mode "sum_delta": min(w_max, floor + (I_t + I_prev) * (|I_t - I_prev| + delta_bias) / denom).
    mode "imax": current-frame intensity only. spatial_kernel > 1 then makes
    0.5 * (I_t + MaxPool). No floor and no change term.
    spatial_kernel > 1 replaces w with 0.5 * (w + MaxPool). temporal_radius > 0 then
    replaces w with 0.5 * (w + max over +/- that many frames). normalize_mean divides
    by the batch mean after both mixes.
    """
    if hist_last.dim() == 4:
        hist_last = hist_last[:, -1]
    prev = torch.cat([hist_last[:, None].to(dtype=real.dtype), real[:, :-1]], dim=1)
    kind = str(mode).strip().lower()
    if kind == "abs":
        delta = (real - prev).abs() * float(scale)
        weight = torch.clamp(delta + float(floor), max=float(w_max))
    elif kind == "sum_delta":
        if float(denom) == 0.0:
            raise ValueError("change weight denom must be non-zero")
        cur = real.float() * float(scale)
        prv = prev.float() * float(scale)
        weight = float(floor) + (cur + prv) * ((cur - prv).abs() + float(delta_bias)) / float(denom)
        weight = torch.clamp(weight, max=float(w_max)).to(dtype=real.dtype)
    elif kind == "imax":
        weight = (real.float() * float(scale)).to(dtype=real.dtype)
    else:
        raise ValueError(f"unknown change weight mode {mode}")
    weight = _spread_spatial_max(weight, spatial_kernel)
    weight = _spread_temporal_max(weight, temporal_radius)
    if normalize_mean:
        weight = _div_mean(weight)
    return weight.detach()


def maxpool_mse(
    real: torch.Tensor,
    fake_list: list[torch.Tensor],
    *,
    kernel_size: int = 5,
    stride: int = 2,
) -> torch.Tensor:
    """Plain MSE after MaxPool. No rain-intensity weight. Mean over the ensemble."""
    pad = kernel_size // 2

    def pool(x: torch.Tensor) -> torch.Tensor:
        b, t, h, w = x.shape
        return F.max_pool2d(x.reshape(b * t, 1, h, w), kernel_size, stride, pad)

    real_q = pool(real)
    return torch.stack([(pool(fake) - real_q).square().mean() for fake in fake_list]).mean()


def _ring_coeff(
    kernel_size: int,
    device,
    dtype,
    *,
    metric: str = "chebyshev",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center=1, d=1 → 0.5, d=2 → 0.25. Returns (coeff, keep_mask) of shape (k, k)."""
    c0 = kernel_size // 2
    yy, xx = torch.meshgrid(
        torch.arange(kernel_size, device=device),
        torch.arange(kernel_size, device=device),
        indexing="ij",
    )
    metric = str(metric).strip().lower()
    if metric in ("manhattan", "l1", "diamond"):
        dist = (yy - c0).abs() + (xx - c0).abs()
        keep = dist <= 2
    else:
        dist = torch.maximum((yy - c0).abs(), (xx - c0).abs())
        keep = torch.ones(kernel_size, kernel_size, device=device, dtype=torch.bool)
    coeff = torch.ones(kernel_size, kernel_size, device=device, dtype=dtype)
    coeff = torch.where(dist == 1, coeff.new_full((), 0.5), coeff)
    coeff = torch.where(dist == 2, coeff.new_full((), 0.25), coeff)
    if metric not in ("manhattan", "l1", "diamond"):
        coeff = torch.where(dist >= 2, coeff.new_full((), 0.25), coeff)
    return coeff, keep


def _chebyshev_ring_coeff(kernel_size: int, device, dtype) -> torch.Tensor:
    coeff, _ = _ring_coeff(kernel_size, device, dtype, metric="chebyshev")
    return coeff


def ring_softmax_pool(
    x: torch.Tensor,
    *,
    kernel_size: int = 5,
    stride: int = 1,
    tau: float = 16.0,
    metric: str = "chebyshev",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft pool: softmax((c⊙x)/τ) as weights, original x as values.

    Replicate-pad so a uniform field stays uniform (zero-pad would mix in 0s).
    Manhattan metric keeps only the 13-cell diamond (L1≤2) inside a 5×5 unfold.
    x: (N, 1, H, W). Returns (pooled, mean entropy over windows).
    """
    pad = kernel_size // 2
    n, _, h, w = x.shape
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    patches = F.unfold(x_pad, kernel_size=kernel_size, padding=0, stride=stride)
    k2 = kernel_size * kernel_size
    coeff, keep = _ring_coeff(kernel_size, x.device, x.dtype, metric=metric)
    coeff = coeff.reshape(1, k2, 1)
    keep = keep.reshape(1, k2, 1)
    logits = (coeff * patches) / max(float(tau), 1e-6)
    logits = logits.masked_fill(~keep, torch.finfo(x.dtype).min)
    weights = torch.softmax(logits, dim=1)
    pooled = (weights * patches).sum(dim=1)
    hp, wp = h + 2 * pad, w + 2 * pad
    oh = (hp - kernel_size) // stride + 1
    ow = (wp - kernel_size) // stride + 1
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1).mean()
    return pooled.view(n, 1, oh, ow), entropy


def ring_softmax_pool_mse(
    real: torch.Tensor,
    fake_list: list[torch.Tensor],
    *,
    kernel_size: int = 5,
    stride: int = 1,
    tau: float = 16.0,
    metric: str = "chebyshev",
) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE after ring-softmax pool. Returns (loss, mean weight entropy)."""

    def pool(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, h, w = x.shape
        return ring_softmax_pool(
            x.reshape(b * t, 1, h, w),
            kernel_size=kernel_size,
            stride=stride,
            tau=tau,
            metric=metric,
        )

    real_q, _ = pool(real)
    losses = []
    ents = []
    for fake in fake_list:
        fq, ent = pool(fake)
        losses.append((fq - real_q).square().mean())
        ents.append(ent)
    return torch.stack(losses).mean(), torch.stack(ents).mean()


def sample_patches(
    real: torch.Tensor,
    *,
    n_frames: int = 4,
    n_biased: int = 4,
    n_uniform: int = 4,
    patch: int = 16,
    temperature: float = 32.0,
) -> dict[str, torch.Tensor]:
    """Pick frames and 16x16 windows. Weights come from the real patch mean and are detached.

    real: (B, T, H, W). Returns times (F,), ys/xs (F, B, P), weight (F, B, P).
    """
    if real.dim() != 4:
        raise ValueError(f"Expected (B,T,H,W), got {tuple(real.shape)}")
    b, t, h, w = real.shape
    if h < patch or w < patch:
        raise ValueError(f"patch {patch} does not fit {h}x{w}")
    n_frames = min(int(n_frames), t)
    n_patch = int(n_biased) + int(n_uniform)
    times = torch.randperm(t, device=real.device)[:n_frames]
    ys = []
    xs = []
    weights = []
    out_w = w - patch + 1
    for ti in times.tolist():
        frame = real[:, ti]
        score = F.avg_pool2d(frame.unsqueeze(1), kernel_size=patch, stride=1).flatten(1)
        prob = score.clamp_min(0) + 1e-6
        prob = prob / prob.sum(dim=1, keepdim=True)
        biased = torch.multinomial(prob, int(n_biased), replacement=False)
        uniform_prob = torch.ones_like(prob)
        uniform_prob.scatter_(1, biased, 0)
        uniform_prob = uniform_prob / uniform_prob.sum(dim=1, keepdim=True)
        uniform = torch.multinomial(uniform_prob, int(n_uniform), replacement=False)
        idx = torch.cat([biased, uniform], dim=1)
        y = torch.div(idx, out_w, rounding_mode="floor")
        x = idx - y * out_w
        crops = _crop(frame, y, x, patch)
        mean = crops.detach().mean(dim=(-1, -2))
        weight = torch.softmax(mean / float(temperature), dim=1)
        ys.append(y)
        xs.append(x)
        weights.append(weight)
    return {
        "times": times,
        "ys": torch.stack(ys, dim=0),
        "xs": torch.stack(xs, dim=0),
        "weight": torch.stack(weights, dim=0),
        "n_patch": torch.tensor(n_patch),
    }


def _crop(frame: torch.Tensor, y: torch.Tensor, x: torch.Tensor, patch: int) -> torch.Tensor:
    """frame (B,H,W), y/x (B,P) -> (B,P,patch,patch). Advanced index keeps grad to frame."""
    b = frame.shape[0]
    dy = torch.arange(patch, device=frame.device)
    dx = torch.arange(patch, device=frame.device)
    yy = y.unsqueeze(-1) + dy
    xx = x.unsqueeze(-1) + dx
    b_idx = torch.arange(b, device=frame.device)[:, None, None, None]
    return frame[b_idx, yy[:, :, :, None], xx[:, :, None, :]]


def sample_patch_clips(
    real: torch.Tensor,
    *,
    clip_len: int = 4,
    n_biased: int = 4,
    n_uniform: int = 4,
    patch: int = 16,
    temperature: float = 32.0,
) -> dict[str, torch.Tensor]:
    """One random consecutive clip; 16x16 windows shared across those frames.

    Weights come from the real clip mean over time and space and are detached.
    real: (B, T, H, W). Returns times (C,), ys/xs/weight (B, P).
    """
    if real.dim() != 4:
        raise ValueError(f"Expected (B,T,H,W), got {tuple(real.shape)}")
    b, t, h, w = real.shape
    if h < patch or w < patch:
        raise ValueError(f"patch {patch} does not fit {h}x{w}")
    clip_len = min(int(clip_len), t)
    start = int(torch.randint(0, t - clip_len + 1, (1,), device=real.device).item())
    times = torch.arange(start, start + clip_len, device=real.device)
    n_patch = int(n_biased) + int(n_uniform)
    out_w = w - patch + 1
    score_map = real[:, start:start + clip_len].mean(dim=1)
    score = F.avg_pool2d(score_map.unsqueeze(1), kernel_size=patch, stride=1).flatten(1)
    prob = score.clamp_min(0) + 1e-6
    prob = prob / prob.sum(dim=1, keepdim=True)
    biased = torch.multinomial(prob, int(n_biased), replacement=False)
    uniform_prob = torch.ones_like(prob)
    uniform_prob.scatter_(1, biased, 0)
    uniform_prob = uniform_prob / uniform_prob.sum(dim=1, keepdim=True)
    uniform = torch.multinomial(uniform_prob, int(n_uniform), replacement=False)
    idx = torch.cat([biased, uniform], dim=1)
    y = torch.div(idx, out_w, rounding_mode="floor")
    x = idx - y * out_w
    crops = []
    for ti in times.tolist():
        crops.append(_crop(real[:, ti], y, x, patch))
    clip = torch.stack(crops, dim=2)
    mean = clip.detach().mean(dim=(-1, -2, -3))
    weight = torch.softmax(mean / float(temperature), dim=1)
    return {
        "times": times,
        "ys": y,
        "xs": x,
        "weight": weight,
        "n_patch": torch.tensor(n_patch),
        "clip_len": torch.tensor(clip_len),
    }


def take_clips(field: torch.Tensor, spec: dict[str, torch.Tensor], patch: int = 16) -> torch.Tensor:
    """field (B,T,H,W) -> (B,P,C,patch,patch) at one spatial window per clip."""
    out = []
    y = spec["ys"]
    x = spec["xs"]
    for ti in spec["times"].tolist():
        out.append(_crop(field[:, ti], y, x, patch))
    return torch.stack(out, dim=2)


def weighted_clip_bce(
    disc: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    spec: dict[str, torch.Tensor],
    *,
    patch: int = 16,
    for_generator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted BCE on (clip, 16, 16) cubes. Weights sum to 1 over patches."""
    weight = spec["weight"]
    fake_p = take_clips(fake, spec, patch)
    b, p, c, h, w = fake_p.shape
    fake_logit = disc(fake_p.reshape(b * p, c, h, w)).reshape(b, p)
    if for_generator:
        loss = (weight * _bce_per_patch(fake_logit, 1.0)).sum(dim=-1).mean()
        return loss, fake_logit.detach().mean(), fake_logit.detach().mean()
    real_p = take_clips(real, spec, patch)
    real_logit = disc(real_p.reshape(b * p, c, h, w)).reshape(b, p)
    loss = (weight * (_bce_per_patch(real_logit, 1.0) + _bce_per_patch(fake_logit, 0.0))).sum(dim=-1).mean()
    return loss, real_logit.detach().mean(), fake_logit.detach().mean()


def take_patches(field: torch.Tensor, spec: dict[str, torch.Tensor], patch: int = 16) -> torch.Tensor:
    """field (B,T,H,W) -> (F,B,P,patch,patch) at the sampled coordinates."""
    out = []
    times = spec["times"]
    for i, ti in enumerate(times.tolist()):
        out.append(_crop(field[:, ti], spec["ys"][i], spec["xs"][i], patch))
    return torch.stack(out, dim=0)


def weighted_patch_bce(
    disc: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    spec: dict[str, torch.Tensor],
    *,
    patch: int = 16,
    for_generator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted BCE. Weights sum to 1 inside each frame, then the frames are averaged.

    Returns (loss, real_logit, fake_logit). real_logit/fake_logit are detached means.
    For the generator, fake must still require grad and the real term is unused.
    """
    weight = spec["weight"]
    fake_p = take_patches(fake, spec, patch)
    fake_logit = disc(fake_p.reshape(-1, patch, patch)).reshape(fake_p.shape[:3])
    if for_generator:
        loss = (weight * _bce_per_patch(fake_logit, 1.0)).sum(dim=-1).mean()
        return loss, fake_logit.detach().mean(), fake_logit.detach().mean()
    real_p = take_patches(real, spec, patch)
    real_logit = disc(real_p.reshape(-1, patch, patch)).reshape(real_p.shape[:3])
    loss = (weight * (_bce_per_patch(real_logit, 1.0) + _bce_per_patch(fake_logit, 0.0))).sum(dim=-1).mean()
    return loss, real_logit.detach().mean(), fake_logit.detach().mean()


def _bce_per_patch(logit: torch.Tensor, label: float) -> torch.Tensor:
    target = torch.full_like(logit, float(label))
    return F.binary_cross_entropy_with_logits(logit, target, reduction="none")
