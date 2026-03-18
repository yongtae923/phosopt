"""
Bayesian optimisation baseline (vimplant method).

Matches the original van Hoof & Lozano procedure:
  - Same physics pipeline (create_grid -> implant_grid -> phosphene map via basecode/electphos)
  - Same cost: (1 - DC) + (1 - 0.1*Y) + Hellinger + penalty when grid invalid
  - DC with percentile threshold (dc_percentile=50), get_yield, hellinger_distance from basecode
  - x0=(0, 0, 20, 25), LHS initial points, custom_stopper (N=5, delta=0.2, thresh=0.05)

When data_dir and hemisphere are provided, uses this vimplant pipeline. Otherwise
falls back to the generic simulator_fn path (for compatibility with experiments
that pass a single simulator).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import cook_initial_point_generator

from .base import (
    BaseOptimizer,
    BudgetConfig,
    OptimizeResult,
)
from .vimplant_cost import target_to_density_1000, vimplant_cost


def custom_stopper(res: Any, N: int = 5, delta: float = 0.2, thresh: float = 0.05) -> bool | None:
    """
    Same as basecode: stop when the best N cost values are within delta ratio
    and the best cost is below thresh.
    """
    if len(res.func_vals) < N:
        return None
    func_vals = np.sort(res.func_vals)
    worst = func_vals[N - 1]
    best = func_vals[0]
    return bool((abs((best - worst) / (worst + 1e-12)) < delta) and (best < thresh))


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimisation over 4 implant params (all electrodes ON)."""

    name = "bayesian"

    def optimize(
        self,
        target_map: np.ndarray,
        simulator_fn: Any,
        budget: BudgetConfig,
        seed: int = 42,
        data_dir: Path | str | None = None,
        hemisphere: str = "LH",
    ) -> OptimizeResult:
        dimensions = [
            Integer(-90, 90, name="alpha"),
            Integer(-15, 110, name="beta"),
            Integer(0, 40, name="offset_from_base"),
            Integer(10, 40, name="shank_length"),
        ]
        x0 = (0, 0, 20, 25)  # basecode initial values
        electrode_mask = np.ones(1000, dtype=np.float32)

        use_vimplant = data_dir is not None and str(data_dir) != ""

        if use_vimplant:
            return self._optimize_vimplant(
                target_map=target_map,
                data_dir=Path(data_dir),
                hemisphere=hemisphere.upper(),
                dimensions=dimensions,
                x0=x0,
                budget=budget,
                seed=seed,
            )

        # Fallback: generic simulator path (existing behavior)
        return self._optimize_generic(
            target_map=target_map,
            simulator_fn=simulator_fn,
            dimensions=dimensions,
            electrode_mask=electrode_mask,
            budget=budget,
            seed=seed,
        )

    def _optimize_vimplant(
        self,
        target_map: np.ndarray,
        data_dir: Path,
        hemisphere: str,
        dimensions: list,
        x0: tuple[int, ...],
        budget: BudgetConfig,
        seed: int,
    ) -> OptimizeResult:
        from simulator.physics_forward import (
            WINDOWSIZE,
            make_phosphene_map_with_contacts,
        )

        target_density = target_to_density_1000(target_map, window_size=WINDOWSIZE)
        score_history: list[float] = []
        call_count = 0
        best_cost = float("inf")
        best_params = np.array(x0, dtype=np.float32)
        best_dice, best_yield, best_hell = 0.0, 0.0, 0.0
        t0 = time.perf_counter()

        def objective(x: list[int]) -> float:
            nonlocal call_count, best_cost, best_params, best_dice, best_yield, best_hell
            call_count += 1
            if time.perf_counter() - t0 > budget.max_wall_clock_sec:
                return 3.0

            alpha, beta, offset_from_base, shank_length = x
            phosphene_map, contacts_xyz_moved, grid_valid, good_coords = (
                make_phosphene_map_with_contacts(
                    data_dir=data_dir,
                    alpha=float(alpha),
                    beta=float(beta),
                    offset_from_base=float(offset_from_base),
                    shank_length=float(shank_length),
                    hemisphere=hemisphere,
                    as_density=True,
                )
            )
            cost, dice, grid_yield, hell_d = vimplant_cost(
                phosphene_map=phosphene_map,
                target_density=target_density,
                contacts_xyz=contacts_xyz_moved,
                good_coords=good_coords,
                grid_valid=grid_valid,
            )
            score = 2.0 - cost  # S = 2 - loss
            score_history.append(score)

            if cost < best_cost:
                best_cost = cost
                best_params = np.array(x, dtype=np.float32)
                best_dice, best_yield, best_hell = dice, grid_yield, hell_d

            return cost

        def callback(res: Any) -> bool:
            if call_count >= budget.max_simulator_calls:
                return True
            if time.perf_counter() - t0 > budget.max_wall_clock_sec:
                return True
            stop = custom_stopper(res, N=5, delta=0.2, thresh=0.05)
            return bool(stop)

        n_calls = min(budget.max_simulator_calls, 150)
        n_initial_points = 10
        lhs = cook_initial_point_generator("lhs", criterion="maximin")

        gp_minimize(
            objective,
            dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            initial_point_generator=lhs,
            random_state=seed,
            x0=[list(x0)],
            callback=[callback],
            n_jobs=1,
        )

        wall = time.perf_counter() - t0
        score_final = 2.0 - best_cost

        all_on_mask = np.ones(1000, dtype=np.float32)
        return OptimizeResult(
            best_params=best_params,
            best_electrode_mask=all_on_mask,
            best_score=score_final,
            dc=best_dice,
            y=best_yield,
            hd=best_hell,
            simulator_calls=call_count,
            wall_clock_time=wall,
            score_history=score_history,
            converged=best_cost < 0.05,
        )

    def _optimize_generic(
        self,
        target_map: np.ndarray,
        simulator_fn: Any,
        dimensions: list,
        electrode_mask: np.ndarray,
        budget: BudgetConfig,
        seed: int,
    ) -> OptimizeResult:
        from .base import EarlyStopper, _sim_call_numpy
        from metrics.eval_metrics import evaluate_all

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
            recon = _sim_call_numpy(
                simulator_fn, params, electrode_mask, target_map.shape
            )
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
        n_calls = min(budget.max_simulator_calls, 300)
        n_initial = min(10, n_calls // 3)
        gp_minimize(
            objective,
            dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial,
            initial_point_generator=lhs,
            random_state=seed,
            callback=[callback],
        )

        wall = time.perf_counter() - t0
        final_recon = _sim_call_numpy(
            simulator_fn, best_params, electrode_mask, target_map.shape
        )
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
            converged=bool(score_history and stopper.update(score_history[-1])),
        )
