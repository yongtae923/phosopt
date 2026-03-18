"""
All-on optimisation baseline.

All 1000 electrodes are activated; only the 4 implant-placement parameters
are optimised via Bayesian optimisation (same engine as the main baseline,
but the electrode mask is fixed to all-ones).

This is structurally identical to ``BayesianOptimizer`` but kept as a
separate class so the benchmark runner can dispatch by name.
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


class AllOnOptimizer(BaseOptimizer):
    """Optimise 4 params with all electrodes ON."""

    name = "all_on"

    def optimize(
        self,
        target_map: np.ndarray,
        simulator_fn: Any,
        budget: BudgetConfig,
        seed: int = 42,
    ) -> OptimizeResult:
        dimensions = [
            Integer(-90, 90, name="alpha"),
            Integer(-15, 110, name="beta"),
            Integer(0, 40, name="offset_from_base"),
            Integer(10, 40, name="shank_length"),
        ]
        electrode_mask = np.ones(1000, dtype=np.float32)

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
            recon = _sim_call_numpy(simulator_fn, params, electrode_mask, target_map.shape)
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
        final_recon = _sim_call_numpy(simulator_fn, best_params, electrode_mask, target_map.shape)
        final_m = evaluate_all(final_recon, target_map)

        return OptimizeResult(
            best_params=best_params,
            best_electrode_mask=electrode_mask,
            best_score=final_m["score"],
            dc=final_m["dc"],
            y=final_m["y"],
            hd=final_m["hd"],
            simulator_calls=call_count,
            wall_clock_time=wall,
            score_history=score_history,
            converged=stopper.update(best_score) if score_history else False,
        )
