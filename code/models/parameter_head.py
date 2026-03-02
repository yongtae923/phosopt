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


class ParameterHead(nn.Module):
    """
    Multi-head predictor:
      - 4 bounded continuous implant parameters
      - 1000 electrode logits
    """

    def __init__(
        self,
        latent_dim: int,
        electrode_dim: int = 1000,
        bounds: ParameterBounds | None = None,
    ) -> None:
        super().__init__()
        self.bounds = bounds or ParameterBounds()
        self.param_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
        )
        self.electrode_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, electrode_dim),
        )

    def _bound_params(self, raw_params: torch.Tensor) -> torch.Tensor:
        # sigmoid + affine enforces physical parameter range.
        low, high = self.bounds.as_tensor(raw_params.device)
        unit = torch.sigmoid(raw_params)
        return low + (high - low) * unit

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_params = self.param_head(z)
        params = self._bound_params(raw_params)
        electrode_logits = self.electrode_head(z)
        return params, electrode_logits
