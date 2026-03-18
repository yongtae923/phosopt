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
        bounds: ParameterBounds | None = None,
    ) -> None:
        super().__init__()
        self.bounds = bounds or ParameterBounds()

        # Raw (pre-sigmoid) shared implant parameters — learned globally
        self._shared_params_raw = nn.Parameter(torch.zeros(4))

        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
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
