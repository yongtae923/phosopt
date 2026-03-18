"""
Abstract interface shared by all per-target optimisation baselines.

Every baseline must implement ``optimize()`` which returns a standardised
result dict that the benchmark runners can log directly.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class BudgetConfig:
    """Budget constraints from exp.md Section 7."""

    max_simulator_calls: int = 300
    max_wall_clock_sec: float = 1200.0   # 20 min default
    patience_calls: int = 30
    min_improvement: float = 1e-4


@dataclass
class OptimizeResult:
    """Standardised result from a single per-target optimisation run."""

    best_params: np.ndarray          # [4]  (alpha, beta, offset, shank)
    best_electrode_mask: np.ndarray  # [1000]  float in [0, 1]
    best_score: float
    dc: float
    y: float
    hd: float
    simulator_calls: int
    wall_clock_time: float
    score_history: list[float]
    converged: bool


class BaseOptimizer(ABC):
    """Common interface for per-target baselines."""

    name: str = "base"

    @abstractmethod
    def optimize(
        self,
        target_map: np.ndarray,
        simulator_fn: Any,
        budget: BudgetConfig,
        seed: int = 42,
    ) -> OptimizeResult:
        ...


def _sim_call_numpy(
    simulator_fn: Any,
    params_np: np.ndarray,
    electrode_mask: np.ndarray,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Call simulator with numpy arrays; return (H, W) map.

    If the simulator output resolution differs from *target_size* the map
    is resized (area interpolation) so that metric functions get matching shapes.
    """
    device = torch.device("cpu")
    if hasattr(simulator_fn, "parameters"):
        try:
            device = next(simulator_fn.parameters()).device
        except StopIteration:
            pass

    p = torch.from_numpy(params_np.astype(np.float32)).unsqueeze(0).to(device)
    logits = torch.from_numpy(
        np.where(electrode_mask > 0.5, 8.0, -8.0).astype(np.float32)
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        out = simulator_fn(p, logits)
    arr = out.squeeze().cpu().numpy()

    if target_size is not None and arr.shape != target_size:
        from skimage.transform import resize
        arr = resize(arr, target_size, preserve_range=True).astype(np.float32)
        mx = arr.max()
        if mx > 0:
            arr /= mx
    return arr


class EarlyStopper:
    """Track score history and decide when to stop."""

    def __init__(self, patience: int, min_improvement: float) -> None:
        self.patience = patience
        self.min_improvement = min_improvement
        self._scores: list[float] = []
        self._best: float = -float("inf")

    def update(self, score: float) -> bool:
        """Return True if optimisation should stop."""
        self._scores.append(score)
        if score > self._best + self.min_improvement:
            self._best = score
        n = len(self._scores)
        if n < self.patience:
            return False
        recent = self._scores[-self.patience:]
        return (max(recent) - min(recent)) < self.min_improvement
