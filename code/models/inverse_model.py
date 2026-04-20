# D:\yongtae\phosopt\code\models\inverse_model.py

"""
Inverse model for PhosOpt self-supervised learning.

This module defines the `InverseModel` class, which consists of:
- A shared implant parameter vector (alpha, beta, offset, shank_length) that is
    learned globally across all images.
- An Encoder that processes input phosphene maps into a latent representation.
- An ElectrodeHead that predicts per-image electrode activations from the latent
    representation.

The forward pass returns both the shared implant parameters and the per-image
    electrode logits, which are then used by the simulator to reconstruct the
    phosphene map for loss computation.
"""

from __future__ import annotations

import torch
from torch import nn

from .encoder import Encoder
from .parameter_head import ElectrodeHead, ParameterBounds


class InverseModel(nn.Module):
    """Shared-implant inverse model.

    The 4 implant placement parameters (alpha, beta, offset, shank_length)
    are a single ``nn.Parameter`` shared across ALL images — they represent
    one physical implant that is fixed once inserted.

    Per-image electrode activations are predicted by Encoder -> ElectrodeHead.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 256,
        electrode_dim: int = 1000,
        map_size: int = 128,
        bounds: ParameterBounds | None = None,
    ) -> None:
        super().__init__()
        self.bounds = bounds or ParameterBounds()

        # Raw (pre-sigmoid) shared implant parameters — learned globally
        self._shared_params_raw = nn.Parameter(torch.zeros(4))

        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim, map_size=map_size)
        self.electrode_head = ElectrodeHead(
            latent_dim=latent_dim,
            electrode_dim=electrode_dim,
        )

    @property
    def shared_params(self) -> torch.Tensor:
        """Bounded shared implant parameters [4]."""
        low, high = self.bounds.as_tensor(self._shared_params_raw.device)
        return low + (high - low) * torch.sigmoid(self._shared_params_raw)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.shared_params.unsqueeze(0).expand(x.size(0), -1)
        z = self.encoder(x)
        electrode_logits = self.electrode_head(z)
        return params, electrode_logits
