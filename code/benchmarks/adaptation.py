"""
Adaptation benchmark runner (exp.md Section 3.3).

Compares how quickly each method adapts to **new, unseen** targets:
  - Bayesian from scratch
  - PhosOpt fine-tune  (start from pre-trained shared model)
  - PhosOpt zero-shot  (no additional optimisation)
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from torch.optim import Adam

from baselines.base import BudgetConfig, EarlyStopper
from logger import ExperimentLog, ExperimentLogger
from metrics.eval_metrics import evaluate_all


def _bayesian_from_scratch(
    target_map: np.ndarray,
    simulator_fn: Any,
    budget: BudgetConfig,
    seed: int,
    data_dir: Path | str | None = None,
    hemisphere: str = "LH",
) -> ExperimentLog:
    from baselines.bayesian import BayesianOptimizer
    opt = BayesianOptimizer()
    result = opt.optimize(
        target_map, simulator_fn, budget, seed,
        data_dir=data_dir, hemisphere=hemisphere,
    )
    active = int((result.best_electrode_mask > 0.5).sum())
    return ExperimentLog(
        benchmark_type="adaptation",
        method="bayesian_scratch",
        seed=seed,
        score=result.best_score,
        loss=2.0 - result.best_score,
        dc=result.dc, y=result.y, hd=result.hd,
        active_electrode_count=active,
        simulator_calls=result.simulator_calls,
        wall_clock_time=result.wall_clock_time,
        training_cost=0.0,
        solving_cost=result.wall_clock_time,
    )


def _phosopt_zeroshot(
    model: Any,
    simulator: Any,
    target_map: np.ndarray,
    seed: int,
) -> ExperimentLog:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(target_map).unsqueeze(0).unsqueeze(0).float().to(device)
        params, logits = model(t)
        recon = simulator(params, logits)
        recon_np = recon.squeeze().cpu().numpy()
    metrics = evaluate_all(recon_np, target_map)
    active = int((torch.sigmoid(logits).squeeze().cpu().numpy() > 0.5).sum())
    return ExperimentLog(
        benchmark_type="adaptation",
        method="phosopt_zeroshot",
        seed=seed,
        score=metrics["score"],
        loss=metrics["loss"],
        dc=metrics["dc"], y=metrics["y"], hd=metrics["hd"],
        active_electrode_count=active,
        simulator_calls=0,
        wall_clock_time=0.0,
        training_cost=0.0,
        solving_cost=0.0,
    )


def _phosopt_finetune(
    model: Any,
    simulator: Any,
    target_map: np.ndarray,
    budget: BudgetConfig,
    seed: int,
    lr: float = 1e-3,
) -> ExperimentLog:
    """Fine-tune electrode logits (and optionally the encoder) on a single new target."""
    device = next(model.parameters()).device
    ft_model = copy.deepcopy(model).to(device)
    ft_model.train()
    torch.manual_seed(seed)

    target_t = torch.from_numpy(target_map).unsqueeze(0).unsqueeze(0).float().to(device)
    opt = Adam(ft_model.parameters(), lr=lr)

    stopper = EarlyStopper(budget.patience_calls, budget.min_improvement)
    call_count = 0
    best_score = -float("inf")
    t0 = time.perf_counter()

    while call_count < budget.max_simulator_calls:
        if time.perf_counter() - t0 > budget.max_wall_clock_sec:
            break

        params, logits = ft_model(target_t)
        recon = simulator(params, logits)
        call_count += 1

        mse = torch.mean((recon - target_t) ** 2)
        opt.zero_grad()
        mse.backward()
        opt.step()

        with torch.no_grad():
            recon_np = recon.squeeze().cpu().numpy()
            metrics = evaluate_all(recon_np, target_map)
            if metrics["score"] > best_score:
                best_score = metrics["score"]
            if stopper.update(metrics["score"]):
                break

    wall = time.perf_counter() - t0

    ft_model.eval()
    with torch.no_grad():
        params, logits = ft_model(target_t)
        recon = simulator(params, logits)
        recon_np = recon.squeeze().cpu().numpy()
    final_m = evaluate_all(recon_np, target_map)
    active = int((torch.sigmoid(logits).squeeze().cpu().numpy() > 0.5).sum())

    return ExperimentLog(
        benchmark_type="adaptation",
        method="phosopt_finetune",
        seed=seed,
        score=final_m["score"],
        loss=final_m["loss"],
        dc=final_m["dc"], y=final_m["y"], hd=final_m["hd"],
        active_electrode_count=active,
        simulator_calls=call_count,
        wall_clock_time=wall,
        training_cost=0.0,
        solving_cost=wall,
    )


def run_adaptation_benchmark(
    test_targets: dict[str, np.ndarray],
    pretrained_model: Any,
    simulator: Any,
    seeds: list[int],
    budget: BudgetConfig,
    output_dir: str | Path,
    experiment_id: str = "adaptation",
    data_dir: Path | str | None = None,
    hemisphere: str = "LH",
) -> list[ExperimentLog]:
    """Run adaptation benchmark: Bayesian scratch vs PhosOpt fine-tune vs zero-shot."""

    out = Path(output_dir)
    logger = ExperimentLogger(out / "adaptation_results.jsonl")
    all_logs: list[ExperimentLog] = []

    total = len(test_targets) * len(seeds) * 3
    done = 0

    for target_id, target_map in test_targets.items():
        for seed in seeds:
            # 1) Bayesian from scratch
            done += 1
            print(f"[{done}/{total}] bayesian_scratch | {target_id} | seed={seed}")
            log_b = _bayesian_from_scratch(
                target_map, simulator, budget, seed,
                data_dir=data_dir, hemisphere=hemisphere,
            )
            log_b.experiment_id = experiment_id
            log_b.target_id = target_id
            logger.log(log_b)
            all_logs.append(log_b)
            print(f"  -> score={log_b.score:.4f}")

            # 2) PhosOpt fine-tune
            done += 1
            print(f"[{done}/{total}] phosopt_finetune | {target_id} | seed={seed}")
            log_ft = _phosopt_finetune(pretrained_model, simulator, target_map, budget, seed)
            log_ft.experiment_id = experiment_id
            log_ft.target_id = target_id
            logger.log(log_ft)
            all_logs.append(log_ft)
            print(f"  -> score={log_ft.score:.4f}")

            # 3) PhosOpt zero-shot
            done += 1
            print(f"[{done}/{total}] phosopt_zeroshot | {target_id} | seed={seed}")
            log_zs = _phosopt_zeroshot(pretrained_model, simulator, target_map, seed)
            log_zs.experiment_id = experiment_id
            log_zs.target_id = target_id
            logger.log(log_zs)
            all_logs.append(log_zs)
            print(f"  -> score={log_zs.score:.4f}")

    print(f"\nAdaptation benchmark complete: {len(all_logs)} entries logged.")
    return all_logs
