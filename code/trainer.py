from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from loss.losses import LossConfig, build_losses, dice_score, hellinger_distance


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-3
    grad_clip_norm: float = 1.0
    allow_nondiff_training: bool = False
    refinement_steps: int = 0
    refinement_lr: float = 1e-2


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_refinement(
    simulator: nn.Module,
    params: torch.Tensor,
    electrode_logits: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    lr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if steps <= 0:
        return params, electrode_logits

    p = params.detach().clone().requires_grad_(True)
    e = electrode_logits.detach().clone().requires_grad_(True)
    opt = Adam([p, e], lr=lr)
    for _ in range(steps):
        recon = simulator(p, e)
        loss = torch.mean((recon - target) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return p.detach(), e.detach()


def train_inverse_model(
    model: nn.Module,
    simulator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_config: LossConfig,
    train_config: TrainConfig,
    valid_electrode_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    dev = _device()
    model = model.to(dev)
    simulator = simulator.to(dev)

    if not getattr(simulator, "is_differentiable", False) and not train_config.allow_nondiff_training:
        raise RuntimeError(
            "Simulator is non-differentiable. "
            "Use differentiable simulator or set allow_nondiff_training=True for debugging only."
        )

    opt = Adam(model.parameters(), lr=train_config.lr)
    history: dict[str, list[float]] = {
        "train_total": [],
        "train_recon": [],
        "val_mse": [],
        "val_dice": [],
        "val_hellinger": [],
    }

    for epoch in range(train_config.epochs):
        model.train()
        total_acc = 0.0
        recon_acc = 0.0
        steps = 0

        train_iter = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch + 1}/{train_config.epochs}",
            leave=False,
        )
        for batch in train_iter:
            target = batch.to(dev)
            params, electrode_logits = model(target)

            if train_config.refinement_steps > 0:
                params, electrode_logits = _run_refinement(
                    simulator=simulator,
                    params=params,
                    electrode_logits=electrode_logits,
                    target=target,
                    steps=train_config.refinement_steps,
                    lr=train_config.refinement_lr,
                )

            recon = simulator(params, electrode_logits)
            losses = build_losses(
                recon=recon,
                target=target,
                params=params,
                electrode_logits=electrode_logits,
                valid_electrode_mask=valid_electrode_mask,
                epoch_idx=epoch,
                config=loss_config,
            )
            loss = losses["total"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if train_config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
            opt.step()

            total_acc += float(losses["total"].item())
            recon_acc += float(losses["recon"].item())
            steps += 1
            train_iter.set_postfix(
                total=f"{losses['total'].item():.4e}",
                recon=f"{losses['recon'].item():.4e}",
            )

        history["train_total"].append(total_acc / max(steps, 1))
        history["train_recon"].append(recon_acc / max(steps, 1))

        val = evaluate_inverse_model(model=model, simulator=simulator, data_loader=val_loader, device=dev)
        history["val_mse"].append(val["mse"])
        history["val_dice"].append(val["dice"])
        history["val_hellinger"].append(val["hellinger"])
        if dev.type == "cuda":
            mem_alloc = torch.cuda.memory_allocated(dev) / (1024**2)
            mem_reserved = torch.cuda.memory_reserved(dev) / (1024**2)
            print(
                f"[Epoch {epoch + 1}/{train_config.epochs}] "
                f"train_total={history['train_total'][-1]:.4e}, "
                f"val_mse={val['mse']:.4e}, val_dice={val['dice']:.4f}, "
                f"cuda_mem_alloc={mem_alloc:.1f}MiB, cuda_mem_reserved={mem_reserved:.1f}MiB"
            )
        else:
            print(
                f"[Epoch {epoch + 1}/{train_config.epochs}] "
                f"train_total={history['train_total'][-1]:.4e}, "
                f"val_mse={val['mse']:.4e}, val_dice={val['dice']:.4f} (CPU)"
            )

    return history


@torch.no_grad()
def evaluate_inverse_model(
    model: nn.Module,
    simulator: nn.Module,
    data_loader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, float]:
    dev = device or _device()
    model.eval()

    mse_sum = 0.0
    dice_sum = 0.0
    h_sum = 0.0
    n = 0
    for batch in tqdm(data_loader, desc="Eval", leave=False):
        target = batch.to(dev)
        params, electrode_logits = model(target)
        recon = simulator(params, electrode_logits)
        mse_sum += float(torch.mean((recon - target) ** 2).item())
        dice_sum += float(dice_score(recon, target).item())
        h_sum += float(hellinger_distance(recon, target).item())
        n += 1

    denom = max(n, 1)
    return {"mse": mse_sum / denom, "dice": dice_sum / denom, "hellinger": h_sum / denom}


@torch.no_grad()
def evaluate_random_baseline(simulator: nn.Module, data_loader: DataLoader) -> dict[str, float]:
    """Random baseline for protocol comparison."""
    dev = _device()
    mse_sum = 0.0
    n = 0
    for batch in data_loader:
        target = batch.to(dev)
        b = target.shape[0]
        params = torch.rand((b, 4), device=dev)
        electrode_logits = torch.randn((b, 1000), device=dev)
        recon = simulator(params, electrode_logits)
        mse_sum += float(torch.mean((recon - target) ** 2).item())
        n += 1
    return {"mse": mse_sum / max(n, 1)}


@torch.no_grad()
def evaluate_four_param_baseline(model: nn.Module, simulator: nn.Module, data_loader: DataLoader) -> dict[str, float]:
    """
    4-params-only baseline:
    uses model-predicted params but sets all electrodes active.
    """
    dev = _device()
    model.eval()
    mse_sum = 0.0
    n = 0
    for batch in data_loader:
        target = batch.to(dev)
        params, _ = model(target)
        electrode_logits = torch.ones((target.shape[0], 1000), device=dev) * 8.0
        recon = simulator(params, electrode_logits)
        mse_sum += float(torch.mean((recon - target) ** 2).item())
        n += 1
    return {"mse": mse_sum / max(n, 1)}
