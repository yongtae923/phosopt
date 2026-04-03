# D:\yongtae\phosopt\code\loss\losses.py

"""
Loss functions for PhosOpt inverse model training.
This module defines:
- `LossConfig`: Configuration for weighting auxiliary loss components.
- `build_losses`: Function to compute total loss as:
    L_main = 2 - dc_soft - 0.1*y + hd
    L_total = L_main + λs*sparsity + λi*invalid_region

Where:
- dc_soft = 1 - soft_dice_loss(recon, target) [soft Dice score]
- y = contact yield from simulator
- hd = Hellinger distance between recon and target
- sparsity = mean electrode activation probability
- invalid_region = penalty for activations in invalid electrode locations
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    sparsity_weight: float = 1e-3
    invalid_region_weight: float = 1e-3
    warmup_epochs: int = 10


def linear_warmup_scale(epoch_idx: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, (epoch_idx + 1) / warmup_epochs))


def soft_dice_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable soft Dice loss (1 - Dice). Works on continuous [0,1] values."""
    flat_p = pred.flatten(start_dim=1)
    flat_t = target.flatten(start_dim=1)
    inter = (flat_p * flat_t).sum(dim=1)
    union = flat_p.sum(dim=1) + flat_t.sum(dim=1)
    dice = (2.0 * inter + eps) / (union + eps)
    return (1.0 - dice).mean()


def dice_score(
    pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.05, eps: float = 1e-8,
) -> torch.Tensor:
    """Hard Dice for evaluation only."""
    pred_bin = (pred > thresh).float()
    target_bin = (target > thresh).float()
    inter = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3))
    return ((2.0 * inter + eps) / (union + eps)).mean()


def hellinger_distance(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    p = pred.clamp_min(eps)
    q = target.clamp_min(eps)

    # Normalize each sample to probability distributions (sum to 1).
    p_sum = p.flatten(start_dim=1).sum(dim=1).view(-1, 1, 1, 1)
    q_sum = q.flatten(start_dim=1).sum(dim=1).view(-1, 1, 1, 1)
    p = p / (p_sum + eps)
    q = q / (q_sum + eps)

    return torch.sqrt(
        torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2, dim=(1, 2, 3)) / 2.0
    ).mean()


def y_metric_from_params(simulator: torch.nn.Module, params: torch.Tensor) -> torch.Tensor:
    """Estimate contact yield as mean soft-validity over contacts."""
    alpha, beta, offset, shank = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    positioned = simulator._create_positioned_grid(alpha, beta, offset, shank)
    _, validity = simulator._soft_prf_lookup(positioned)
    return validity.mean()


def kl_divergence(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    p = pred.clamp_min(eps)
    q = target.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=(1, 2, 3)).mean()


def build_losses(
    recon: torch.Tensor,
    target: torch.Tensor,
    params: torch.Tensor,
    electrode_logits: torch.Tensor,
    simulator: torch.nn.Module,
    valid_electrode_mask: torch.Tensor | None,
    epoch_idx: int,
    config: LossConfig,
) -> dict[str, torch.Tensor]:
    """Compute vimplant-objective-based loss with auxiliary regularization terms.
    
    Args:
        recon: Reconstructed phosphene map [B,1,H,W]
        target: Target phosphene map [B,1,H,W]
        params: Implant parameters [B,4]
        electrode_logits: Electrode logits [B,1000]
        simulator: DifferentiableSimulator for y metric
        valid_electrode_mask: Electrode validity mask or None
        epoch_idx: Current epoch for warmup scheduling
        config: Loss configuration
    
    Returns:
        Dictionary with loss components and total loss.
    """
    # Main objective: 2 - dc_soft - 0.1*y + hd
    dc_soft = 1.0 - soft_dice_loss(recon, target)
    y = y_metric_from_params(simulator, params)
    hd = hellinger_distance(recon, target)
    loss_main = 2.0 - dc_soft - 0.1 * y + hd

    # Auxiliary regularization terms
    electrode_prob = torch.sigmoid(electrode_logits)
    sparsity_loss = electrode_prob.mean()

    if valid_electrode_mask is None:
        invalid_region_loss = torch.zeros((), device=recon.device)
    else:
        invalid_mask = (1.0 - valid_electrode_mask).to(recon.device)
        invalid_region_loss = (electrode_prob * invalid_mask).mean()

    # Apply warmup to auxiliary terms
    warm = linear_warmup_scale(epoch_idx=epoch_idx, warmup_epochs=config.warmup_epochs)
    total = (
        loss_main
        + warm * config.sparsity_weight * sparsity_loss
        + warm * config.invalid_region_weight * invalid_region_loss
    )
    
    return {
        "total": total,
        "loss_main": loss_main,
        "dc_soft": dc_soft,
        "y_metric": y,
        "hd_metric": hd,
        "sparsity": sparsity_loss,
        "invalid_region": invalid_region_loss,
    }
