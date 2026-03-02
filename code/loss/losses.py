from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    recon_weight: float = 1.0
    sparsity_weight: float = 1e-3
    param_prior_weight: float = 1e-4
    invalid_region_weight: float = 1e-3
    warmup_epochs: int = 10


def linear_warmup_scale(epoch_idx: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, (epoch_idx + 1) / warmup_epochs))


def dice_score(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.01, eps: float = 1e-8) -> torch.Tensor:
    pred_bin = (pred > thresh).float()
    target_bin = (target > thresh).float()
    inter = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3))
    return ((2.0 * inter + eps) / (union + eps)).mean()


def hellinger_distance(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = pred.clamp_min(eps)
    q = target.clamp_min(eps)
    return torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2, dim=(1, 2, 3)) / 2.0).mean()


def kl_divergence(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = pred.clamp_min(eps)
    q = target.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=(1, 2, 3)).mean()


def build_losses(
    recon: torch.Tensor,
    target: torch.Tensor,
    params: torch.Tensor,
    electrode_logits: torch.Tensor,
    valid_electrode_mask: torch.Tensor | None,
    epoch_idx: int,
    config: LossConfig,
) -> dict[str, torch.Tensor]:
    recon_loss = F.mse_loss(recon, target)

    electrode_prob = torch.sigmoid(electrode_logits)
    sparsity_loss = electrode_prob.mean()

    # Parameters are already bounded; weak center prior improves identifiability.
    param_center = params.mean(dim=0, keepdim=True)
    param_prior_loss = ((params - param_center) ** 2).mean()

    if valid_electrode_mask is None:
        invalid_region_loss = torch.zeros((), device=recon.device)
    else:
        invalid_mask = (1.0 - valid_electrode_mask).to(recon.device)
        invalid_region_loss = (electrode_prob * invalid_mask).mean()

    warm = linear_warmup_scale(epoch_idx=epoch_idx, warmup_epochs=config.warmup_epochs)
    total = (
        config.recon_weight * recon_loss
        + warm * config.sparsity_weight * sparsity_loss
        + warm * config.param_prior_weight * param_prior_loss
        + warm * config.invalid_region_weight * invalid_region_loss
    )
    return {
        "total": total,
        "recon": recon_loss,
        "sparsity": sparsity_loss,
        "param_prior": param_prior_loss,
        "invalid_region": invalid_region_loss,
    }
