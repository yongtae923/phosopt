# D:\yongtae\phosopt\code\trainer.py

"""
Training script for PhosOpt inverse model.

Given:
- a differentiable simulator,
- a training dataset of target phosphene maps,
- and a model architecture for the inverse mapping,
    this script trains the inverse model to predict implant parameters and 
    electrode activations that reconstruct the target maps via the simulator.

The training loop includes:
1) Forward pass through the model and simulator,
2) Loss computation with multiple components,
3) Backpropagation and optimization,
4) Validation and checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from loss.losses import LossConfig, build_losses, dice_score, hellinger_distance


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8
    lr: float = 3e-4
    shared_params_lr: float = 1e-2
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    allow_nondiff_training: bool = False
    refinement_steps: int = 0
    refinement_lr: float = 1e-2
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    min_epochs_for_early_stop: int = 20
    patience_for_early_stop: int = 5


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
    """Refine only electrode_logits per-image; params are shared and untouched."""
    if steps <= 0:
        return params, electrode_logits

    from torch.optim import Adam
    e = electrode_logits.detach().clone().requires_grad_(True)
    opt = Adam([e], lr=lr)
    p_detached = params.detach()
    for _ in range(steps):
        recon = simulator(p_detached, e)
        loss = torch.mean((recon - target) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return params, e.detach()


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    history: dict[str, list[float]],
    best_val_total_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_val_total_loss": best_val_total_loss,
        },
        path,
    )


def load_checkpoint(
    path: Path, model: nn.Module, device: torch.device,
) -> dict[str, Any]:
    """Load checkpoint and return metadata. Model weights are restored in-place."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def train_inverse_model(
    model: nn.Module,
    simulator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_config: LossConfig,
    train_config: TrainConfig,
    valid_electrode_mask: torch.Tensor | None = None,
    checkpoint_dir: Path | None = None,
    resume_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dev = _device()
    model = model.to(dev)
    simulator = simulator.to(dev)

    if not getattr(simulator, "is_differentiable", False) and not train_config.allow_nondiff_training:
        raise RuntimeError(
            "Simulator is non-differentiable. "
            "Use differentiable simulator or set allow_nondiff_training=True for debugging only."
        )

    shared_raw = [model._shared_params_raw] if hasattr(model, "_shared_params_raw") else []
    if shared_raw:
        network_params = [p for p in model.parameters() if p is not model._shared_params_raw]
    else:
        network_params = list(model.parameters())
    opt = AdamW(
        [
            {"params": network_params, "lr": train_config.lr},
            {"params": shared_raw, "lr": train_config.shared_params_lr, "weight_decay": 0.0},
        ] if shared_raw else [{"params": network_params, "lr": train_config.lr}],
        weight_decay=train_config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        opt,
        mode="min",
        factor=train_config.scheduler_factor,
        patience=train_config.scheduler_patience,
    )

    # Defaults
    start_epoch = 0
    best_val_total_loss = float("inf")
    history: dict[str, list[float]] = {
        "train_total": [],
        "train_main": [],
        "val_total_loss": [],
        "val_mse": [],
        "val_dice": [],
        "val_hellinger": [],
    }

    # Restore from checkpoint
    if resume_checkpoint is not None:
        start_epoch = resume_checkpoint["epoch"] + 1
        history = resume_checkpoint["history"]
        best_val_total_loss = resume_checkpoint.get(
            "best_val_total_loss",
            resume_checkpoint.get("best_val_mse", float("inf")),
        )
        if "train_main" not in history and "train_recon" in history:
            history["train_main"] = history["train_recon"]
        opt.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        print(f"[Resume] Continuing from epoch {start_epoch + 1}/{train_config.epochs} "
              f"(best_val_total_loss={best_val_total_loss:.4e})")

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"[EarlyStopping] Min epochs: {train_config.min_epochs_for_early_stop}, "
          f"Patience: {train_config.patience_for_early_stop}")

    # Early stopping tracking
    epochs_without_improvement = 0

    for epoch in range(start_epoch, train_config.epochs):
        model.train()
        total_acc = 0.0
        main_acc = 0.0
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
                simulator=simulator,
                valid_electrode_mask=valid_electrode_mask,
                config=loss_config,
            )
            loss = losses["total"]

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if steps == 0:
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5
                elec_prob = torch.sigmoid(electrode_logits)
                lr_now = opt.param_groups[0]["lr"]
                sp = model.shared_params.detach().cpu() if hasattr(model, "shared_params") else None
                sp_str = (
                    f"a={sp[0]:.1f} b={sp[1]:.1f} o={sp[2]:.1f} s={sp[3]:.1f}"
                    if sp is not None else "N/A"
                )
                print(
                    f"  [diag] params=[{sp_str}] "
                    f"target:[{target.min():.3f},{target.max():.3f}] "
                    f"recon:[{recon.min():.3f},{recon.max():.3f}] "
                    f"grad={grad_norm:.2e} lr={lr_now:.1e} "
                    f"loss_main={losses['loss_main'].item():.4e} "
                    f"dc_soft={losses['dc_soft'].item():.4f} "
                    f"y={losses['y_metric'].item():.4f} "
                    f"hd={losses['hd_metric'].item():.4f}"
                )

            if train_config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
            opt.step()

            total_acc += float(losses["total"].item())
            main_acc += float(losses["loss_main"].item())
            steps += 1
            train_iter.set_postfix(
                total=f"{losses['total'].item():.4e}",
                main=f"{losses['loss_main'].item():.4e}",
            )

        avg_total = total_acc / max(steps, 1)
        avg_main = main_acc / max(steps, 1)
        history["train_total"].append(avg_total)
        history["train_main"].append(avg_main)

        val = evaluate_inverse_model(
            model=model, simulator=simulator, data_loader=val_loader, loss_config=loss_config, device=dev,
        )
        history["val_total_loss"].append(val["total_loss"])
        history["val_mse"].append(val["mse"])
        history["val_dice"].append(val["dice"])
        history["val_hellinger"].append(val["hellinger"])

        scheduler.step(val["total_loss"])

        # Track best and early stopping (based on val_total_loss)
        is_best = val["total_loss"] < best_val_total_loss
        if is_best:
            best_val_total_loss = val["total_loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        sp = model.shared_params.detach().cpu() if hasattr(model, "shared_params") else None
        sp_str = (
            f"[a={sp[0]:.1f} b={sp[1]:.1f} o={sp[2]:.1f} s={sp[3]:.1f}]"
            if sp is not None else ""
        )

        if dev.type == "cuda":
            mem = torch.cuda.memory_allocated(dev) / (1024**2)
            print(
                f"[Epoch {epoch + 1}/{train_config.epochs}] "
                f"loss={avg_total:.4e} main={avg_main:.4e} "
                f"val_total_loss={val['total_loss']:.4e} val_mse={val['mse']:.4e} val_dice={val['dice']:.4f} "
                f"params={sp_str} cuda={mem:.0f}MiB"
                f"{' *best*' if is_best else ''}"
            )
        else:
            print(
                f"[Epoch {epoch + 1}/{train_config.epochs}] "
                f"loss={avg_total:.4e} main={avg_main:.4e} "
                f"val_total_loss={val['total_loss']:.4e} val_mse={val['mse']:.4e} val_dice={val['dice']:.4f} "
                f"params={sp_str} (CPU)"
                f"{' *best*' if is_best else ''}"
            )

        # Save checkpoint every epoch
        if checkpoint_dir is not None:
            ckpt_path = checkpoint_dir / "checkpoint_latest.pt"
            _save_checkpoint(ckpt_path, epoch, model, opt, scheduler, history, best_val_total_loss)
            if is_best:
                best_path = checkpoint_dir / "checkpoint_best.pt"
                _save_checkpoint(best_path, epoch, model, opt, scheduler, history, best_val_total_loss)
                print(f"  [ckpt] Saved best checkpoint (val_total_loss={best_val_total_loss:.4e})")

        # Early stopping check
        if epoch + 1 >= train_config.min_epochs_for_early_stop and \
           epochs_without_improvement >= train_config.patience_for_early_stop:
            print(f"[EarlyStopping] Stopped at epoch {epoch + 1} "
                  f"(no improvement for {epochs_without_improvement} epochs)")
            break

    return history


@torch.no_grad()
def evaluate_inverse_model(
    model: nn.Module,
    simulator: nn.Module,
    data_loader: DataLoader,
    loss_config: LossConfig,
    device: torch.device | None = None,
) -> dict[str, float]:
    dev = device or _device()
    model.eval()

    total_loss_sum = 0.0
    mse_sum = 0.0
    dice_sum = 0.0
    h_sum = 0.0
    n = 0
    for batch in tqdm(data_loader, desc="Eval", leave=False):
        target = batch.to(dev)
        params, electrode_logits = model(target)
        recon = simulator(params, electrode_logits)
        
        losses = build_losses(
            recon=recon,
            target=target,
            params=params,
            electrode_logits=electrode_logits,
            simulator=simulator,
            valid_electrode_mask=None,
            config=loss_config,
        )
        
        total_loss_sum += float(losses["total"].item())
        mse_sum += float(torch.mean((recon - target) ** 2).item())
        dice_sum += float(dice_score(recon, target).item())
        h_sum += float(hellinger_distance(recon, target).item())
        n += 1

    denom = max(n, 1)
    return {
        "total_loss": total_loss_sum / denom,
        "mse": mse_sum / denom,
        "dice": dice_sum / denom,
        "hellinger": h_sum / denom,
    }


@torch.no_grad()
def evaluate_random_baseline(simulator: nn.Module, data_loader: DataLoader) -> dict[str, float]:
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
def evaluate_four_param_baseline(
    model: nn.Module, simulator: nn.Module, data_loader: DataLoader,
) -> dict[str, float]:
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
