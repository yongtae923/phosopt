# D:\yongtae\phosopt\code\models\encoder.py

"""
Encoder module for PhosOpt inverse model.

This module defines an E2E-style convolutional encoder adapted for 256x256 
    phosphene maps.

The architecture consists of:
- A series of Conv-BN-LeakyReLU blocks with downsampling via MaxPool2
- Residual blocks for feature refinement
- A final convolution to reduce channels to 1
- A fully connected head to produce a latent vector for downstream parameter 
    prediction
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn


def _conv_block(
    in_ch: int,
    out_ch: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    pool: nn.Module | None = None,
) -> List[nn.Module]:
    """Conv-BN-LeakyReLU block, optionally followed by a pooling layer."""
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(inplace=True),
    ]
    if pool is not None:
        layers.append(pool)
    return layers


class ResidualBlock(nn.Module):
    """Conv-BN-LeakyReLU x2 with skip connection (E2E style)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class Encoder(nn.Module):
    """
    E2E-style encoder adapted for 256x256 phosphene maps.

    Based on the basecode E2E_Encoder, scaled up for 256x256 input
    and modified to output a latent vector (instead of raw electrode
    amplitudes) for downstream ParameterHead consumption.

    Spatial path:
      256 --[conv]--> 256 --[pool]--> 128 --[pool]--> 64 --[pool]--> 32
      --> ResBlock xN --> conv reduce --> flatten(32x32=1024) --> Linear --> latent_dim

    Channel path:
      in_channels -> c1 -> c2 -> c3 -> c4 -> (ResBlock xN) -> c4/2 -> 1
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 256,
        input_map_size: int = 256,
        stage_channels: Sequence[int] = (16, 32, 64, 128),
        num_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        if len(stage_channels) != 4:
            raise ValueError(f"stage_channels must have 4 entries, got {len(stage_channels)}")
        if num_res_blocks < 1:
            raise ValueError(f"num_res_blocks must be >= 1, got {num_res_blocks}")
        if input_map_size < 8 or input_map_size % 8 != 0:
            raise ValueError(
                f"input_map_size must be divisible by 8 and >= 8, got {input_map_size}"
            )

        c1, c2, c3, c4 = (int(c) for c in stage_channels)
        if min(c1, c2, c3, c4) < 1:
            raise ValueError(f"all stage channels must be >= 1, got {stage_channels}")

        c_mid = max(c4 // 2, 1)
        final_hw = input_map_size // 8
        fc_in_features = final_hw * final_hw
        res_blocks = [ResidualBlock(c4) for _ in range(num_res_blocks)]

        self.features = nn.Sequential(
            *_conv_block(in_channels, c1),
            *_conv_block(c1, c2, pool=nn.MaxPool2d(2)),
            *_conv_block(c2, c3, pool=nn.MaxPool2d(2)),
            *_conv_block(c3, c4, pool=nn.MaxPool2d(2)),
            *res_blocks,
            *_conv_block(c4, c_mid),
            nn.Conv2d(c_mid, 1, 3, padding=1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in_features, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.head(h)
