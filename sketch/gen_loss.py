"""Generative losses: BCE adversarial + low-weight pixel L2."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rain_weight(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(1.0 + x, max=24.0)


def weighted_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    w = rain_weight(target)
    return (torch.abs(target - pred) * w).mean()


def pool_regularization(
    real: torch.Tensor,
    fake_list: list[torch.Tensor],
    *,
    kernel_size: int = 5,
    stride: int = 2,
) -> torch.Tensor:
    """J_pool = L_wdis(Q(real), mean_i Q(fake_i)), Q = MaxPool2d(k=5, s=2)."""
    pad = kernel_size // 2
    real_q = F.max_pool2d(real, kernel_size=kernel_size, stride=stride, padding=pad)
    fake_q = torch.stack(
        [
            F.max_pool2d(f, kernel_size=kernel_size, stride=stride, padding=pad)
            for f in fake_list
        ],
        dim=0,
    ).mean(dim=0)
    return weighted_l1(fake_q, real_q)


def bce_logits(logit: torch.Tensor, label: float) -> torch.Tensor:
    target = torch.full_like(logit, float(label))
    return F.binary_cross_entropy_with_logits(logit, target)


def discriminator_loss(real_logit: torch.Tensor, fake_logit: torch.Tensor) -> torch.Tensor:
    return bce_logits(real_logit, 1.0) + bce_logits(fake_logit, 0.0)


def adversarial_loss(fake_logit: torch.Tensor) -> torch.Tensor:
    return bce_logits(fake_logit, 1.0)


def pixel_l2(real: torch.Tensor, fake_list: list[torch.Tensor]) -> torch.Tensor:
    """Mean squared error on [0, 128], averaged over the noise ensemble."""
    return torch.stack([(fake - real).square().mean() for fake in fake_list]).mean()
