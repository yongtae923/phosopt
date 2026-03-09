from __future__ import annotations

import torch
from torch import nn

from .encoder import Encoder
from .parameter_head import ParameterBounds, ParameterHead


class InverseModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 256,
        electrode_dim: int = 1000,
        bounds: ParameterBounds | None = None,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.head = ParameterHead(
            latent_dim=latent_dim,
            electrode_dim=electrode_dim,
            bounds=bounds,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.head(z)
