# D:\yongtae\phosopt\code\models\parameter_head.py

"""
Parameter head module for PhosOpt inverse model.

This module defines:
- `ParameterBounds`: A dataclass to specify min/max bounds for implant 
    parameters.
- `ContinuousHead`: A head to predict 4 bounded continuous implant parameters.
- `ElectrodeHead`: A head to predict 1000 electrode logits (soft weights).
- `ParameterHead`: A combined head that wraps both ContinuousHead and 
    ElectrodeHead for backward compatibility.

The continuous parameters are bounded using a sigmoid transformation to ensure 
    they stay within the specified physical limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ParameterBounds:
    alpha_min: float = -90.0
    alpha_max: float = 90.0
    beta_min: float = -15.0
    beta_max: float = 110.0
    offset_min: float = 0.0
    offset_max: float = 40.0
    shank_min: float = 10.0
    shank_max: float = 40.0

    def as_tensor(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        low = torch.tensor(
            [self.alpha_min, self.beta_min, self.offset_min, self.shank_min],
            dtype=torch.float32,
            device=device,
        )
        high = torch.tensor(
            [self.alpha_max, self.beta_max, self.offset_max, self.shank_max],
            dtype=torch.float32,
            device=device,
        )
        return low, high


class ContinuousHead(nn.Module):
    """Predicts 4 bounded continuous implant parameters."""

    def __init__(self, latent_dim: int, bounds: ParameterBounds | None = None) -> None:
        super().__init__()
        self.bounds = bounds or ParameterBounds()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.net(z)
        low, high = self.bounds.as_tensor(raw.device)
        return low + (high - low) * torch.sigmoid(raw)


class ElectrodeHead(nn.Module):
    """Predicts 1000 electrode logits (soft weights)."""

    def __init__(self, latent_dim: int, electrode_dim: int = 1000) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, electrode_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# Backward-compatible alias
class ParameterHead(nn.Module):
    """Combined head wrapping ContinuousHead + ElectrodeHead."""

    def __init__(
        self,
        latent_dim: int,
        electrode_dim: int = 1000,
        bounds: ParameterBounds | None = None,
    ) -> None:
        super().__init__()
        self.continuous = ContinuousHead(latent_dim, bounds)
        self.electrode = ElectrodeHead(latent_dim, electrode_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.continuous(z), self.electrode(z)
