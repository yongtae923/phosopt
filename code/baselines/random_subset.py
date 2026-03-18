"""
Random subset optimisation baseline.

A random subset of electrodes is selected, then the 4 implant-placement
parameters are optimised via Bayesian optimisation.  The random mask is
re-sampled periodically to explore different electrode configurations.
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


class RandomSubsetOptimizer(BaseOptimizer):
    """Optimise 4 params with a randomly chosen electrode subset."""

    name = "random_subset"

    def __init__(self, active_ratio: float = 0.3, n_mask_restarts: int = 5) -> None:
        self.active_ratio = active_ratio
        self.n_mask_restarts = n_mask_restarts

    def _random_mask(self, rng: np.random.Generator) -> np.ndarray:
        mask = np.zeros(1000, dtype=np.float32)
        n_active = max(1, int(1000 * self.active_ratio))
        idx = rng.choice(1000, size=n_active, replace=False)
        mask[idx] = 1.0
        return mask

    def optimize(
        self,
        target_map: np.ndarray,
        simulator_fn: Any,
        budget: BudgetConfig,
        seed: int = 42,
    ) -> OptimizeResult:
        rng = np.random.default_rng(seed)
        dimensions = [
            Integer(-90, 90, name="alpha"),
            Integer(-15, 110, name="beta"),
            Integer(0, 40, name="offset_from_base"),
            Integer(10, 40, name="shank_length"),
        ]

        overall_best_score = -float("inf")
        overall_best_params = np.array([0.0, 0.0, 20.0, 25.0])
        overall_best_mask = np.ones(1000, dtype=np.float32)
        score_history: list[float] = []
        total_calls = 0
        t0 = time.perf_counter()

        calls_per_restart = max(
            10, budget.max_simulator_calls // max(self.n_mask_restarts, 1)
        )

        for restart in range(self.n_mask_restarts):
            if total_calls >= budget.max_simulator_calls:
                break
            if time.perf_counter() - t0 > budget.max_wall_clock_sec:
                break

            mask = self._random_mask(rng)
            stopper = EarlyStopper(budget.patience_calls, budget.min_improvement)
            local_best_score = -float("inf")
            local_best_params = np.array([0.0, 0.0, 20.0, 25.0])
            local_calls = 0

            def objective(x: list[int]) -> float:
                nonlocal total_calls, local_calls, local_best_score, local_best_params
                total_calls += 1
                local_calls += 1
                if time.perf_counter() - t0 > budget.max_wall_clock_sec:
                    return 3.0

                params = np.array(x, dtype=np.float32)
                recon = _sim_call_numpy(simulator_fn, params, mask, target_map.shape)
                metrics = evaluate_all(recon, target_map)
                score_history.append(metrics["score"])

                if metrics["score"] > local_best_score:
                    local_best_score = metrics["score"]
                    local_best_params = params.copy()
                return metrics["loss"]

            def callback(res: Any) -> bool:
                if total_calls >= budget.max_simulator_calls:
                    return True
                if local_calls >= calls_per_restart:
                    return True
                if time.perf_counter() - t0 > budget.max_wall_clock_sec:
                    return True
                return stopper.update(score_history[-1]) if score_history else False

            n_calls = min(calls_per_restart, budget.max_simulator_calls - total_calls)
            if n_calls <= 0:
                break

            lhs = cook_initial_point_generator("lhs", criterion="maximin")
            gp_minimize(
                objective,
                dimensions,
                n_calls=n_calls,
                n_initial_points=min(5, n_calls // 2),
                initial_point_generator=lhs,
                random_state=int(rng.integers(0, 2**31)),
                callback=[callback],
            )

            if local_best_score > overall_best_score:
                overall_best_score = local_best_score
                overall_best_params = local_best_params.copy()
                overall_best_mask = mask.copy()

        wall = time.perf_counter() - t0
        final_recon = _sim_call_numpy(simulator_fn, overall_best_params, overall_best_mask, target_map.shape)
        final_m = evaluate_all(final_recon, target_map)

        return OptimizeResult(
            best_params=overall_best_params,
            best_electrode_mask=overall_best_mask,
            best_score=final_m["score"],
            dc=final_m["dc"],
            y=final_m["y"],
            hd=final_m["hd"],
            simulator_calls=total_calls,
            wall_clock_time=wall,
            score_history=score_history,
            converged=False,
        )
