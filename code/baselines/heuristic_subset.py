"""
Heuristic subset optimisation baselines.

Two strategies for selecting which electrodes to activate:

1. **center-prior**  -- electrodes closest to the target map's centre of mass
2. **intensity-prior** -- electrodes whose receptive-field centres overlap
   high-intensity regions of the target map

After the mask is fixed, the 4 implant-placement parameters are optimised
with Bayesian optimisation (same as all-on / random subset).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import cook_initial_point_generator

from metrics.eval_metrics import evaluate_all
from .base import (
    BaseOptimizer,
    BudgetConfig,
    EarlyStopper,
    OptimizeResult,
    _sim_call_numpy,
)


def _centre_of_mass(target_map: np.ndarray) -> tuple[float, float]:
    total = target_map.sum()
    if total < 1e-12:
        h, w = target_map.shape
        return h / 2.0, w / 2.0
    ys, xs = np.mgrid[0: target_map.shape[0], 0: target_map.shape[1]]
    cy = float((ys * target_map).sum() / total)
    cx = float((xs * target_map).sum() / total)
    return cy, cx


def _center_prior_mask(target_map: np.ndarray, n_electrodes: int = 1000, active_ratio: float = 0.3) -> np.ndarray:
    """Activate electrodes nearest to the target centre of mass.

    Since we don't have electrode-to-pixel mapping at mask-selection time,
    we use a deterministic grid ordering: electrodes are indexed 0..999
    as a 10x10x10 grid.  We rank them by distance of their grid-index
    centroid to the target's normalised centre of mass.
    """
    cy, cx = _centre_of_mass(target_map)
    h, w = target_map.shape
    cy_norm, cx_norm = cy / h, cx / w

    n = int(round(n_electrodes ** (1 / 3)))
    coords = np.array(
        [(i / n, j / n, k / n) for i in range(n) for j in range(n) for k in range(n)],
        dtype=np.float32,
    )
    # project to 2D by using first two grid axes
    dists = np.sqrt((coords[:, 0] - cy_norm) ** 2 + (coords[:, 1] - cx_norm) ** 2)
    n_active = max(1, int(n_electrodes * active_ratio))
    top_idx = np.argsort(dists)[:n_active]
    mask = np.zeros(n_electrodes, dtype=np.float32)
    mask[top_idx] = 1.0
    return mask


def _intensity_prior_mask(target_map: np.ndarray, n_electrodes: int = 1000, active_ratio: float = 0.3) -> np.ndarray:
    """Activate electrodes whose grid positions map to high-intensity regions."""
    h, w = target_map.shape
    n = int(round(n_electrodes ** (1 / 3)))

    intensities = np.zeros(n_electrodes, dtype=np.float32)
    idx = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                py = int(i / n * h) % h
                px = int(j / n * w) % w
                intensities[idx] = target_map[py, px]
                idx += 1

    n_active = max(1, int(n_electrodes * active_ratio))
    top_idx = np.argsort(intensities)[-n_active:]
    mask = np.zeros(n_electrodes, dtype=np.float32)
    mask[top_idx] = 1.0
    return mask


class _HeuristicBase(BaseOptimizer):
    """Shared BO loop for both heuristic strategies."""

    def _get_mask(self, target_map: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def optimize(
        self,
        target_map: np.ndarray,
        simulator_fn: Any,
        budget: BudgetConfig,
        seed: int = 42,
    ) -> OptimizeResult:
        mask = self._get_mask(target_map)
        dimensions = [
            Integer(-90, 90, name="alpha"),
            Integer(-15, 110, name="beta"),
            Integer(0, 40, name="offset_from_base"),
            Integer(10, 40, name="shank_length"),
        ]

        stopper = EarlyStopper(budget.patience_calls, budget.min_improvement)
        score_history: list[float] = []
        call_count = 0
        best_score = -float("inf")
        best_params = np.array([0.0, 0.0, 20.0, 25.0])
        t0 = time.perf_counter()

        def objective(x: list[int]) -> float:
            nonlocal call_count, best_score, best_params
            call_count += 1
            if time.perf_counter() - t0 > budget.max_wall_clock_sec:
                return 3.0
            params = np.array(x, dtype=np.float32)
            recon = _sim_call_numpy(simulator_fn, params, mask, target_map.shape)
            metrics = evaluate_all(recon, target_map)
            score_history.append(metrics["score"])
            if metrics["score"] > best_score:
                best_score = metrics["score"]
                best_params = params.copy()
            return metrics["loss"]

        def callback(res: Any) -> bool:
            if call_count >= budget.max_simulator_calls:
                return True
            if time.perf_counter() - t0 > budget.max_wall_clock_sec:
                return True
            return stopper.update(score_history[-1]) if score_history else False

        lhs = cook_initial_point_generator("lhs", criterion="maximin")
        gp_minimize(
            objective,
            dimensions,
            n_calls=min(budget.max_simulator_calls, 300),
            n_initial_points=min(10, budget.max_simulator_calls // 3),
            initial_point_generator=lhs,
            random_state=seed,
            callback=[callback],
        )

        wall = time.perf_counter() - t0
        final_recon = _sim_call_numpy(simulator_fn, best_params, mask, target_map.shape)
        final_m = evaluate_all(final_recon, target_map)

        return OptimizeResult(
            best_params=best_params,
            best_electrode_mask=mask,
            best_score=final_m["score"],
            dc=final_m["dc"],
            y=final_m["y"],
            hd=final_m["hd"],
            simulator_calls=call_count,
            wall_clock_time=wall,
            score_history=score_history,
            converged=stopper.update(best_score) if score_history else False,
        )


class HeuristicCenterOptimizer(_HeuristicBase):
    name = "heuristic_center"

    def _get_mask(self, target_map: np.ndarray) -> np.ndarray:
        return _center_prior_mask(target_map)


class HeuristicIntensityOptimizer(_HeuristicBase):
    name = "heuristic_intensity"

    def _get_mask(self, target_map: np.ndarray) -> np.ndarray:
        return _intensity_prior_mask(target_map)
