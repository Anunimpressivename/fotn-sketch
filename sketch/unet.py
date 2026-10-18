from __future__ import annotations

import torch
import torch.nn as nn


class ConvBN(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNet(nn.Module):
    """Two-down U-Net. Channels in/out carry time."""

    def __init__(self, in_ch: int, out_ch: int, base: int = 16):
        super().__init__()
        c = base
        self.enc0 = ConvBN(in_ch, c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBN(c, c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBN(c * 2, c * 4))
        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec1 = ConvBN(c * 4, c * 2)
        self.up0 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec0 = ConvBN(c * 2, c)
        self.head = nn.Conv2d(c, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        d1 = self.dec1(torch.cat([self.up1(e2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))
        return self.head(d0)
