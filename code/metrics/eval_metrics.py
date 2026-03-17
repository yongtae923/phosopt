"""
Evaluation metrics for the PhosOpt experiment pipeline.

Implements the vimplant-compatible metrics:
  - DC  (Dice coefficient)
  - Y   (Yield = fraction of electrodes landing in valid cortex)
  - HD  (Hellinger distance between density maps)
  - S   (Composite score = DC + 0.1*Y - HD)
  - Loss = 2 - S
"""
from __future__ import annotations

import numpy as np


def dice_coefficient(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.05,
    empty_score: float = 1.0,
) -> float:
    """Hard Dice coefficient (DC) on binarised maps.

    Matches the basecode ``lossfunc.DC`` definition.
    """
    pred_bin = (pred > threshold).astype(bool)
    target_bin = (target > threshold).astype(bool)
    total = pred_bin.sum() + target_bin.sum()
    if total == 0:
        return empty_score
    intersection = np.logical_and(pred_bin, target_bin).sum()
    return float(2.0 * intersection / total)


def yield_metric(
    contacts_xyz: np.ndarray,
    good_coords: np.ndarray,
) -> float:
    """Fraction of electrode contacts that land in valid cortex voxels.

    Parameters
    ----------
    contacts_xyz : (3, N) array of contact-point coordinates.
    good_coords  : (3, M) array of valid cortex voxel coordinates.

    Returns 0.0 when *contacts_xyz* is empty.
    """
    if contacts_xyz.size == 0:
        return 0.0
    b1 = np.round(contacts_xyz.T).astype(np.int32)
    b2 = good_coords.T.astype(np.int32) if good_coords.ndim == 2 else good_coords.astype(np.int32)
    if b2.ndim == 1:
        b2 = b2.reshape(-1, 3)
    good_set = set(map(tuple, b2))
    hits = sum(1 for row in b1 if tuple(row) in good_set)
    return float(hits / max(b1.shape[0], 1))


def hellinger_distance(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Hellinger distance between two (flattened) distributions.

    Same as basecode lossfunc.hellinger_distance:
      H(p,q) = sqrt(sum((sqrt(p)-sqrt(q))^2)) / sqrt(2)
    Maps are flattened and normalized to sum=1 before computing.
    Returns 0.0 when both maps are zero.
    """
    p = pred.flatten().astype(np.float64) + eps
    q = target.flatten().astype(np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2))


def composite_score(dc: float, y: float, hd: float) -> float:
    """S = DC + 0.1*Y - HD  (vimplant composite score; HD = Hellinger distance)."""
    return dc + 0.1 * y - hd


def composite_loss(score: float) -> float:
    """Loss = 2 - S  (vimplant composite loss)."""
    return 2.0 - score


def evaluate_all(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.05,
    contacts_xyz: np.ndarray | None = None,
    good_coords: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all metrics and return a dict.

    If *contacts_xyz* / *good_coords* are not provided the yield is set to 0.
    """
    dc = dice_coefficient(pred, target, threshold=threshold)
    hd = hellinger_distance(pred, target, eps=1e-12)
    if contacts_xyz is not None and good_coords is not None:
        y = yield_metric(contacts_xyz, good_coords)
    else:
        y = 0.0
    s = composite_score(dc, y, hd)
    return {
        "dc": dc,
        "y": y,
        "hd": hd,
        "score": s,
        "loss": composite_loss(s),
    }
