"""Full-frame discriminator on the 5x5 high-pass of one radar image."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


class GlobalDiscriminator(nn.Module):
    """One logit per 64x64 frame. The first two convolutions stay at stride 1."""

    def __init__(self, in_ch: int = 1, base_c: int = 16, blur_kernel: int = 5):
        super().__init__()
        c = int(base_c)
        self.blur_kernel = int(blur_kernel)
        self.stem = nn.Sequential(
            self._conv(in_ch, c, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            self._conv(c, c, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down = nn.Sequential(
            self._conv(c, c * 2, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            self._conv(c * 2, c * 4, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            self._conv(c * 4, c * 4, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            self._conv(c * 4, c * 4, kernel_size=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = spectral_norm(nn.Linear(c * 4 * 4 * 4, 1))

    @staticmethod
    def _conv(in_c: int, out_c: int, kernel_size: int, stride: int) -> nn.Module:
        return spectral_norm(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected (N,H,W) or (N,1,H,W), got {tuple(x.shape)}")
        k = self.blur_kernel
        blur = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)
        h = self.down(self.stem(x - blur))
        return self.fc(h.flatten(1)).squeeze(1)


def global_bce(
    disc: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    times: torch.Tensor,
    *,
    for_generator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BCE on the same frames the patch loss sampled. Mean over batch and frames."""
    idx = times.long()
    fake_f = fake[:, idx]
    b, n, h, w = fake_f.shape
    fake_logit = disc(fake_f.reshape(b * n, 1, h, w)).reshape(b, n)
    if for_generator:
        loss = _bce(fake_logit, 1.0).mean()
        return loss, fake_logit.detach().mean(), fake_logit.detach().mean()
    real_f = real[:, idx]
    real_logit = disc(real_f.reshape(b * n, 1, h, w)).reshape(b, n)
    loss = (_bce(real_logit, 1.0) + _bce(fake_logit, 0.0)).mean()
    return loss, real_logit.detach().mean(), fake_logit.detach().mean()


def _bce(logit: torch.Tensor, label: float) -> torch.Tensor:
    target = torch.full_like(logit, float(label))
    return F.binary_cross_entropy_with_logits(logit, target, reduction="none")
