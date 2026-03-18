"""
PhosOpt per-target optimisation.

Directly optimises the 4 implant parameters + 1000 electrode logits for
a single target map using gradient descent through the differentiable
simulator.  This is the per-target variant of PhosOpt (as opposed to
the shared-parameter version used in the Generalized benchmark).

Based on ``diag_sim_upper_bound.py`` but formalised as a baseline method.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch.optim import Adam

from metrics.eval_metrics import evaluate_all
from .base import BaseOptimizer, BudgetConfig, EarlyStopper, OptimizeResult


class PhosOptPerTargetOptimizer(BaseOptimizer):
    """Gradient-based per-target optimisation through differentiable simulator."""

    name = "phosopt_per_target"

    def __init__(self, lr: float = 0.05) -> None:
        self.lr = lr

    def optimize(
        self,
        target_map: np.ndarray,
        simulator_fn: Any,
        budget: BudgetConfig,
        seed: int = 42,
    ) -> OptimizeResult:
        torch.manual_seed(seed)
        device = next(
            (p.device for p in simulator_fn.parameters()),
            torch.device("cpu"),
        ) if hasattr(simulator_fn, "parameters") else torch.device("cpu")

        target_h, target_w = target_map.shape

        target_t = (
            torch.from_numpy(target_map.astype(np.float32))
            .unsqueeze(0).unsqueeze(0)
            .to(device)
        )

        params = torch.zeros(1, 4, device=device, requires_grad=True)
        electrode_logits = torch.zeros(1, 1000, device=device, requires_grad=True)
        opt = Adam([params, electrode_logits], lr=self.lr)

        stopper = EarlyStopper(budget.patience_calls, budget.min_improvement)
        score_history: list[float] = []
        call_count = 0
        best_score = -float("inf")
        best_params_np = np.zeros(4, dtype=np.float32)
        best_logits_np = np.zeros(1000, dtype=np.float32)
        t0 = time.perf_counter()

        is_diff = getattr(simulator_fn, "is_differentiable", False)

        while call_count < budget.max_simulator_calls:
            if time.perf_counter() - t0 > budget.max_wall_clock_sec:
                break

            recon = simulator_fn(params, electrode_logits)
            call_count += 1

            # Resize recon if shape mismatch (numpy sim -> 1000x1000)
            if recon.shape[-2:] != target_t.shape[-2:]:
                recon = torch.nn.functional.interpolate(
                    recon, size=(target_h, target_w), mode="bilinear", align_corners=False,
                )

            mse = torch.mean((recon - target_t) ** 2)

            if is_diff:
                opt.zero_grad()
                mse.backward()
                opt.step()
            else:
                # Non-differentiable: use finite-difference style random perturbation
                with torch.no_grad():
                    recon_np = recon.squeeze().cpu().numpy()
                    metrics = evaluate_all(recon_np, target_map)
                    score_history.append(metrics["score"])
                    if metrics["score"] > best_score:
                        best_score = metrics["score"]
                        best_params_np = params.detach().squeeze().cpu().numpy().copy()
                        best_logits_np = electrode_logits.detach().squeeze().cpu().numpy().copy()
                    # Simple random perturbation when gradients are unavailable
                    params.data += torch.randn_like(params) * 0.5
                    electrode_logits.data += torch.randn_like(electrode_logits) * 0.1
                if stopper.update(metrics["score"]):
                    break
                continue

            with torch.no_grad():
                recon_np = recon.squeeze().cpu().numpy()
                metrics = evaluate_all(recon_np, target_map)
                score = metrics["score"]
                score_history.append(score)

                if score > best_score:
                    best_score = score
                    best_params_np = params.detach().squeeze().cpu().numpy().copy()
                    best_logits_np = electrode_logits.detach().squeeze().cpu().numpy().copy()

            if stopper.update(score):
                break

        wall = time.perf_counter() - t0
        best_mask = (1.0 / (1.0 + np.exp(-best_logits_np))).astype(np.float32)

        # Final evaluation
        with torch.no_grad():
            p = torch.from_numpy(best_params_np).unsqueeze(0).to(device)
            l = torch.from_numpy(best_logits_np).unsqueeze(0).to(device)
            final_recon = simulator_fn(p, l).squeeze().cpu().numpy()
        if final_recon.shape != target_map.shape:
            from skimage.transform import resize as sk_resize
            final_recon = sk_resize(final_recon, target_map.shape, preserve_range=True).astype(np.float32)
            mx = final_recon.max()
            if mx > 0:
                final_recon /= mx
        final_m = evaluate_all(final_recon, target_map)

        return OptimizeResult(
            best_params=best_params_np,
            best_electrode_mask=best_mask,
            best_score=final_m["score"],
            dc=final_m["dc"],
            y=final_m["y"],
            hd=final_m["hd"],
            simulator_calls=call_count,
            wall_clock_time=wall,
            score_history=score_history,
            converged=len(score_history) > 0 and stopper.update(score_history[-1]),
        )
