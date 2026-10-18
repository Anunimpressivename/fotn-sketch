"""2D discriminator for a single 16x16 radar patch."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class PatchDiscriminator(nn.Module):
    """Spectral-norm conv tower. Input (N, P, P) or (N, 1, P, P), one logit each.

    Three stride-2 convs: 16→2×2, 8→1×1. Default P=16 matches older checkpoints.
    """

    def __init__(self, in_ch: int = 1, base_c: int = 32, patch_size: int = 16):
        super().__init__()
        p = int(patch_size)
        spatial = p // 8
        if spatial < 1:
            raise ValueError(f"patch_size={p} is too small for three stride-2 blocks")
        self.patch_size = p
        c1 = int(base_c)
        self.net = nn.Sequential(
            self._down(in_ch, c1),
            self._down(c1, c1 * 2),
            self._down(c1 * 2, c1 * 4),
        )
        self.fc = spectral_norm(nn.Linear(c1 * 4 * spatial * spatial, 1))

    @staticmethod
    def _down(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(
                f"Expected (N,{self.patch_size},{self.patch_size}) or "
                f"(N,1,{self.patch_size},{self.patch_size}), got {tuple(x.shape)}"
            )
        if x.shape[-2] != self.patch_size or x.shape[-1] != self.patch_size:
            raise ValueError(
                f"Expected spatial {self.patch_size}x{self.patch_size}, got {tuple(x.shape)}"
            )
        h = self.net(x)
        return self.fc(h.flatten(1)).squeeze(1)


class PatchClipDiscriminator(nn.Module):
    """3D D on a short (clip, 16, 16) cube. One logit per cube.

    The first two convolutions are 3x3x3 with stride 1 so a period-2 lattice
    is visible in space and across adjacent frames. Later blocks downsample
    only spatially.
    """

    def __init__(self, in_ch: int = 1, base_c: int = 16, clip_len: int = 4):
        super().__init__()
        c = int(base_c)
        t = int(clip_len)
        self.clip_len = t
        self.stem = nn.Sequential(
            self._conv(in_ch, c, stride=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            self._conv(c, c, stride=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down = nn.Sequential(
            self._conv(c, c * 2, stride=(1, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            self._conv(c * 2, c * 4, stride=(1, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            self._conv(c * 4, c * 4, stride=(1, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = spectral_norm(nn.Linear(c * 4 * t * 2 * 2, 1))

    @staticmethod
    def _conv(in_c: int, out_c: int, stride: tuple[int, int, int]) -> nn.Module:
        return spectral_norm(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        elif x.dim() != 5:
            raise ValueError(f"Expected (N,T,16,16) or (N,1,T,16,16), got {tuple(x.shape)}")
        if x.shape[2] != self.clip_len:
            raise ValueError(f"Expected clip_len={self.clip_len}, got T={x.shape[2]}")
        h = self.down(self.stem(x))
        return self.fc(h.flatten(1)).squeeze(1)
