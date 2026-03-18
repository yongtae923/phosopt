"""
Vimplant (van Hoof & Lozano) cost and metrics — matches basecode exactly.

Uses basecode lossfunc: DC (Dice with percentile threshold), get_yield,
hellinger_distance. Cost = (1 - a*DC) + (1 - b*Y) + c*Hellinger + penalty
with a=1, b=0.1, c=1 and penalty 0.25 per term when grid is invalid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASECODE_DIR = PROJECT_ROOT / "basecode"
if str(BASECODE_DIR) not in sys.path:
    sys.path.insert(0, str(BASECODE_DIR))

from lossfunc import DC, get_yield, hellinger_distance


DC_PERCENTILE = 50  # basecode dc_percentile
PENALTY = 0.25      # basecode penalty when grid invalid
A, B, C = 1.0, 0.1, 1.0  # loss_comb (1, 0.1, 1) dice, yield, hellinger


def target_to_density_1000(target_map: np.ndarray, window_size: int = 1000) -> np.ndarray:
    """Resize target to window_size x window_size and normalize to density (max then sum)."""
    if target_map.shape[0] == window_size and target_map.shape[1] == window_size:
        density = target_map.astype(np.float64).copy()
    else:
        from skimage.transform import resize
        density = resize(
            target_map.astype(np.float64),
            (window_size, window_size),
            preserve_range=True,
            anti_aliasing=True,
        )
    density /= max(density.max(), 1e-12)
    density /= max(density.sum(), 1e-12)
    return density.astype(np.float32)


def vimplant_cost(
    phosphene_map: np.ndarray,
    target_density: np.ndarray,
    contacts_xyz: np.ndarray,
    good_coords: np.ndarray,
    grid_valid: bool,
    dc_percentile: int = DC_PERCENTILE,
) -> tuple[float, float, float, float]:
    """
    Compute vimplant cost and components (same formula as basecode f()).

    phosphene_map and target_density must be same shape and normalized to
    density (sum = 1) for Hellinger. good_coords shape (3, M) for get_yield.

    Returns
    -------
    cost : float
    dice : float
    grid_yield : float
    hell_d : float
    """
    bin_thresh = float(np.percentile(target_density, dc_percentile))
    if bin_thresh <= 0:
        bin_thresh = 1e-9

    dice, _im1, _im2 = DC(target_density, phosphene_map, bin_thresh)
    grid_yield = get_yield(contacts_xyz, good_coords)
    hell_d = hellinger_distance(phosphene_map.flatten(), target_density.flatten())

    par1 = 1.0 - (A * dice)
    par2 = 1.0 - (B * grid_yield)
    if np.isnan(hell_d) or np.isinf(hell_d):
        par3 = 1.0
    else:
        par3 = C * hell_d

    if np.isnan(phosphene_map).any() or np.sum(phosphene_map) == 0:
        par1 = 1.0
    if not grid_valid:
        cost = par1 + PENALTY + par2 + PENALTY + par3 + PENALTY
    else:
        cost = par1 + par2 + par3
    if np.isnan(cost) or np.isinf(cost):
        cost = 3.0

    return float(cost), float(dice), float(grid_yield), float(hell_d)
