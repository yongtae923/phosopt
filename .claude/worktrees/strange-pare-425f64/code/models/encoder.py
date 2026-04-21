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

from typing import List

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
    E2E-style encoder adapted for variable input map sizes (e.g., 128x128, 256x256).

    Based on the basecode E2E_Encoder, scaled up for 256x256 input
    and modified to output a latent vector (instead of raw electrode
    amplitudes) for downstream ParameterHead consumption.

    Spatial path (3x MaxPool2d):
      map_size --[conv]--> map_size --[pool]--> map_size/2 --[pool]--> map_size/4 --[pool]--> map_size/8
      --> ResBlock x4 --> conv reduce --> flatten((map_size/8)²) --> Linear --> latent_dim

    Channel path:
      in_channels -> 16 -> 32 -> 64 -> 128 -> (ResBlock x4) -> 64 -> 1
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 256, map_size: int = 128) -> None:
        super().__init__()
        # Compute final spatial dimension after 3x downsampling (2x pooling each)
        final_spatial_size = map_size // 8
        linear_input_size = final_spatial_size * final_spatial_size
        
        self.features = nn.Sequential(
            *_conv_block(in_channels, 16),
            *_conv_block(16, 32, pool=nn.MaxPool2d(2)),
            *_conv_block(32, 64, pool=nn.MaxPool2d(2)),
            *_conv_block(64, 128, pool=nn.MaxPool2d(2)),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            *_conv_block(128, 64),
            nn.Conv2d(64, 1, 3, padding=1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.head(h)
