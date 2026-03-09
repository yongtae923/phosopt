from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Conv-BN-ReLU x2 with skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class DownBlock(nn.Module):
    """Downsample: Conv(stride=2)-BN-ReLU + ResidualBlock."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.res = ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class Encoder(nn.Module):
    """
    ResNet-style encoder for phosphene maps.
    Preserves spatial info through 5 stages before final pooling.
    1×256×256 → 32→64→128→256→256 → AdaptiveAvgPool → Linear → latent_dim
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stage1 = DownBlock(32, 64)     # 64×64
        self.stage2 = DownBlock(64, 128)    # 32×32
        self.stage3 = DownBlock(128, 256)   # 16×16
        self.stage4 = DownBlock(256, 256)   # 8×8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)    # 128×128
        h = self.stage1(h)  # 64×64
        h = self.stage2(h)  # 32×32
        h = self.stage3(h)  # 16×16
        h = self.stage4(h)  # 8×8
        h = self.pool(h).flatten(start_dim=1)
        return self.proj(h)
