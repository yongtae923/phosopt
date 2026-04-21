# D:\yongtae\phosopt\code\simulator\simulator_wrapper.py

"""
Simulator wrapper for PhosOpt inverse model.
This module defines two classes:
1) `SimulatorWrapper`: A wrapper around an already differentiable simulator 
    callable.
2) `NumpySimulatorAdapter`: An adapter for the current numpy-based forward 
    simulator, which is non-differentiable and should not be used for gradient 
    training.

SimulatorWrapper expects a simulator function with the signature:
    (params: Tensor[B,4], electrode_prob: Tensor[B,1000]) -> Tensor[B,1,H,W]
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from simulator.physics_forward import make_phosphene_map


class SimulatorWrapper(nn.Module):
    """
    Wrapper around an already differentiable simulator callable.

    simulator_fn signature:
      (params: Tensor[B,4], electrode_prob: Tensor[B,1000]) -> Tensor[B,1,H,W]
    """

    def __init__(self, simulator_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.simulator_fn = simulator_fn
        self.is_differentiable = True

    def forward(self, params: torch.Tensor, electrode_logits: torch.Tensor) -> torch.Tensor:
        electrode_prob = torch.sigmoid(electrode_logits)
        return self.simulator_fn(params, electrode_prob)


class NumpySimulatorAdapter(nn.Module):
    """
    Inference/baseline adapter for current numpy-based forward simulator.
    This path is non-differentiable and should not be used for gradient training.
    """

    def __init__(self, data_dir: str | Path, hemisphere: str = "LH") -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.hemisphere = hemisphere
        self.is_differentiable = False

    @torch.no_grad()
    def forward(self, params: torch.Tensor, electrode_logits: torch.Tensor) -> torch.Tensor:
        electrode_prob = torch.sigmoid(electrode_logits).detach().cpu().numpy()
        params_np = params.detach().cpu().numpy()
        outputs = []
        for i in range(params_np.shape[0]):
            alpha, beta, offset, shank = params_np[i].tolist()
            out = make_phosphene_map(
                data_dir=self.data_dir,
                alpha=float(alpha),
                beta=float(beta),
                offset_from_base=float(offset),
                shank_length=float(shank),
                electrode_activation=electrode_prob[i].astype(np.float32),
                hemisphere=self.hemisphere,
            )
            outputs.append(out.astype(np.float32))
        arr = np.stack(outputs, axis=0)
        return torch.from_numpy(arr).unsqueeze(1).to(device=params.device)
