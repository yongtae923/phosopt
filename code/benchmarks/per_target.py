"""
Per-target benchmark runner (exp.md Section 3.1).

Each target map is optimised **independently** by each method.
Budget: max 300 simulator calls, max 20 min, early stop if score
improves < 1e-4 over 30 calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from baselines import METHOD_REGISTRY
from baselines.base import BudgetConfig
from logger import ExperimentLog, ExperimentLogger


def run_per_target_benchmark(
    targets: dict[str, np.ndarray],
    simulator_fn: Any,
    method_names: list[str],
    seeds: list[int],
    budget: BudgetConfig,
    output_dir: str | Path,
    experiment_id: str = "per_target",
    data_dir: Path | str | None = None,
    hemisphere: str = "LH",
) -> list[ExperimentLog]:
    """Run all per-target experiments and return collected logs."""

    logger = ExperimentLogger(Path(output_dir) / "per_target_results.jsonl")
    results: list[ExperimentLog] = []

    total = len(targets) * len(method_names) * len(seeds)
    done = 0

    for target_id, target_map in targets.items():
        for method_name in method_names:
            optimizer_cls = METHOD_REGISTRY.get(method_name)
            if optimizer_cls is None:
                print(f"[WARN] Unknown method '{method_name}', skipping.")
                continue

            for seed in seeds:
                done += 1
                print(f"[{done}/{total}] {method_name} | {target_id} | seed={seed}")

                optimizer = optimizer_cls()
                kwargs = dict(
                    target_map=target_map,
                    simulator_fn=simulator_fn,
                    budget=budget,
                    seed=seed,
                )
                if method_name == "bayesian" and data_dir is not None:
                    kwargs["data_dir"] = data_dir
                    kwargs["hemisphere"] = hemisphere
                result = optimizer.optimize(**kwargs)

                active_count = int((result.best_electrode_mask > 0.5).sum())

                log = ExperimentLog(
                    experiment_id=experiment_id,
                    benchmark_type="per_target",
                    method=method_name,
                    seed=seed,
                    target_id=target_id,
                    score=result.best_score,
                    loss=2.0 - result.best_score,
                    dc=result.dc,
                    y=result.y,
                    hd=result.hd,
                    active_electrode_count=active_count,
                    simulator_calls=result.simulator_calls,
                    wall_clock_time=result.wall_clock_time,
                    training_cost=0.0,
                    solving_cost=result.wall_clock_time,
                    extra={
                        "best_params": result.best_params.tolist(),
                        "converged": result.converged,
                    },
                )
                logger.log(log)
                results.append(log)

                print(
                    f"  -> score={result.best_score:.4f} DC={result.dc:.4f} "
                    f"HD={result.hd:.4f} calls={result.simulator_calls} "
                    f"time={result.wall_clock_time:.1f}s"
                )

    print(f"\nPer-target benchmark complete: {len(results)} runs logged.")
    return results
