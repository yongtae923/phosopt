"""
Differentiable phosphene simulator (torch-based).

Approximates physics_forward.py with continuous operations for gradient-based learning.

Key approximations vs numpy version:
  - Voxel matching: soft gaussian kernel instead of exact integer lookup
  - Convex hull ray intersection: pre-computed lookup table with bilinear interpolation
  - Gaussian rendering: chunked differentiable splatting on 2D map
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASECODE_DIR = PROJECT_ROOT / "basecode"
if str(BASECODE_DIR) not in sys.path:
    sys.path.insert(0, str(BASECODE_DIR))

from .physics_forward import (
    N_CONTACTPOINTS_SHANK,
    SPACING_ALONG_XY,
    VIEW_ANGLE,
    AMP,
    load_retinotopy,
)

_DEG2RAD = math.pi / 180.0
_SPREAD = math.sqrt(AMP / 675.0)
_CMAG_A = 0.75
_CMAG_B = 120.0
_CMAG_K = 17.3


def _extract_v1_prf(
    data: dict, hemisphere: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract V1 voxel positions and their PRF properties (angle, ecc, sigma)."""
    v1_coords = data["v1_coords_lh"] if hemisphere == "LH" else data["v1_coords_rh"]
    polar_map, ecc_map, sigma_map = data["polar_map"], data["ecc_map"], data["sigma_map"]

    positions, prf_props = [], []
    for i in range(v1_coords.shape[1]):
        x, y, z = int(v1_coords[0, i]), int(v1_coords[1, i]), int(v1_coords[2, i])
        pol = float(polar_map[x, y, z])
        ecc = float(ecc_map[x, y, z])
        sig = float(sigma_map[x, y, z])
        if ecc > 0 and pol != 0:
            positions.append([x, y, z])
            prf_props.append([pol, ecc, sig])

    return np.array(positions, dtype=np.float32), np.array(prf_props, dtype=np.float32)


def _precompute_surface_distances(
    gm_points: np.ndarray,
    start_location: list[float],
    alpha_range: tuple[float, float] = (-90.0, 90.0),
    beta_range: tuple[float, float] = (-15.0, 110.0),
    n_alpha: int = 37,
    n_beta: int = 26,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute ray-brain surface distances for a grid of (alpha, beta) angles."""
    import trimesh

    mesh = trimesh.points.PointCloud(gm_points).convex_hull
    start = np.array(start_location, dtype=np.float64)

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    betas = np.linspace(beta_range[0], beta_range[1], n_beta)
    distances = np.full((n_alpha, n_beta), 20.0, dtype=np.float32)

    for i, a_deg in enumerate(alphas):
        for j, b_deg in enumerate(betas):
            a = math.radians(a_deg)
            b = math.radians(b_deg)
            # Direction = Rx(alpha) @ Ry(beta) @ [0, 0, -1]
            d = np.array([
                [-math.sin(b)],
                [math.sin(a) * math.cos(b)],
                [-math.cos(a) * math.cos(b)],
            ]).reshape(1, 3)
            try:
                locs, _, _ = mesh.ray.intersects_location(
                    ray_origins=start.reshape(1, 3), ray_directions=d,
                )
                if len(locs) > 0:
                    distances[i, j] = float(np.linalg.norm(locs[0] - start))
            except Exception:
                pass

    return distances, alphas.astype(np.float32), betas.astype(np.float32)


class DifferentiableSimulator(nn.Module):
    """
    Differentiable phosphene simulator for gradient-based inverse learning.

    Pre-computes static subject-specific data (retinotopy, brain geometry) once.
    Forward pass is fully differentiable w.r.t. params and electrode logits.
    """

    def __init__(
        self,
        data_dir: str | Path,
        hemisphere: str = "LH",
        map_size: int = 256,
        soft_match_sigma: float = 1.5,
        render_chunk_size: int = 50,
    ) -> None:
        super().__init__()
        self.hemisphere = hemisphere.upper()
        self.map_size = map_size
        self.soft_match_sigma = soft_match_sigma
        self.render_chunk_size = render_chunk_size
        self.is_differentiable = True

        n = N_CONTACTPOINTS_SHANK
        spacing = SPACING_ALONG_XY

        print(f"[DifferentiableSimulator] Loading retinotopy from {data_dir} ...")
        data = load_retinotopy(Path(data_dir))

        # --- V1 voxel positions and PRF properties ---
        v1_pos, v1_prf = _extract_v1_prf(data, self.hemisphere)
        if v1_pos.shape[0] == 0:
            raise ValueError("No valid V1 voxels found for this hemisphere/subject.")
        self.register_buffer("v1_pos", torch.from_numpy(v1_pos))
        self.register_buffer("v1_prf", torch.from_numpy(v1_prf))
        print(f"  V1 voxels: {v1_pos.shape[0]}")

        # --- Start location ---
        start = data["median_lh"] if self.hemisphere == "LH" else data["median_rh"]
        self.register_buffer("start_loc", torch.tensor(start, dtype=torch.float32))

        # --- Surface distance lookup table ---
        gm = data["gm_lh"] if self.hemisphere == "LH" else data["gm_rh"]
        print("  Pre-computing surface distance lookup table ...")
        surf_dist, alphas, betas = _precompute_surface_distances(gm, start)
        self.register_buffer("surf_dist_lut", torch.from_numpy(surf_dist))
        self.register_buffer("alpha_grid", torch.from_numpy(alphas))
        self.register_buffer("beta_grid", torch.from_numpy(betas))

        # --- Base grid template (ordering: y-slow, x-mid, z-fast = matches original) ---
        y_vals = torch.arange(n, dtype=torch.float32) * spacing
        x_vals = torch.arange(n, dtype=torch.float32) * spacing
        z_fracs = torch.linspace(0, 1, n)
        yy, xx, zz = torch.meshgrid(y_vals, x_vals, z_fracs, indexing="ij")
        template = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        self.register_buffer("grid_template", template)

        print("  Initialization complete.")

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def _rotation_matrix(self, alpha_deg: torch.Tensor, beta_deg: torch.Tensor) -> torch.Tensor:
        """Rx(alpha) @ Ry(beta).  [B] -> [B, 3, 3]"""
        a = alpha_deg * _DEG2RAD
        b = beta_deg * _DEG2RAD
        ca, sa = torch.cos(a), torch.sin(a)
        cb, sb = torch.cos(b), torch.sin(b)
        ones = torch.ones_like(a)
        zeros = torch.zeros_like(a)

        Rx = torch.stack([
            torch.stack([ones, zeros, zeros], -1),
            torch.stack([zeros, ca, -sa], -1),
            torch.stack([zeros, sa, ca], -1),
        ], dim=-2)
        Ry = torch.stack([
            torch.stack([cb, zeros, sb], -1),
            torch.stack([zeros, ones, zeros], -1),
            torch.stack([-sb, zeros, cb], -1),
        ], dim=-2)
        return Rx @ Ry

    def _interpolate_surface_dist(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation into pre-computed surface distance LUT. [B] -> [B]"""
        B = alpha.shape[0]
        a_min, a_max = self.alpha_grid[0], self.alpha_grid[-1]
        b_min, b_max = self.beta_grid[0], self.beta_grid[-1]
        a_norm = 2.0 * (alpha - a_min) / (a_max - a_min + 1e-8) - 1.0
        b_norm = 2.0 * (beta - b_min) / (b_max - b_min + 1e-8) - 1.0
        # grid_sample: input [B,1,Na,Nb], grid [B,1,1,2] with (x=beta, y=alpha)
        grid_coords = torch.stack([b_norm, a_norm], dim=-1).view(B, 1, 1, 2)
        lut = self.surf_dist_lut.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        sampled = F.grid_sample(
            lut, grid_coords, mode="bilinear", padding_mode="border", align_corners=True,
        )
        return sampled.view(B).clamp(min=1.0)

    def _create_positioned_grid(
        self, alpha: torch.Tensor, beta: torch.Tensor,
        offset: torch.Tensor, shank: torch.Tensor,
    ) -> torch.Tensor:
        """Create rotated and positioned contact grid. [B] x4 -> [B, 1000, 3]"""
        B = alpha.shape[0]

        # Scale z by shank_length (out-of-place to preserve autograd graph)
        template = self.grid_template.unsqueeze(0).expand(B, -1, -1)
        xy = template[:, :, :2]
        z_scaled = template[:, :, 2:3] * shank.view(B, 1, 1)
        grid = torch.cat([xy, z_scaled], dim=-1)
        center = grid.mean(dim=1, keepdim=True)
        grid = grid - center

        # Rotate
        R = self._rotation_matrix(alpha, beta)
        rotated = torch.bmm(grid, R.transpose(1, 2))

        # Direction = R @ [0, 0, -1]
        ref = torch.tensor([0.0, 0.0, -1.0], device=alpha.device)
        direction = (R @ ref).squeeze(-1)
        direction = F.normalize(direction, dim=-1)

        # Grid center = start + direction * (surface_dist - shank/2 - offset)
        surf_dist = self._interpolate_surface_dist(alpha, beta)
        penetration = surf_dist - shank / 2.0 - offset
        grid_center = self.start_loc.unsqueeze(0) + direction * penetration.unsqueeze(-1)

        return rotated + grid_center.unsqueeze(1)

    def _soft_prf_lookup(self, contact_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Soft gaussian kernel matching to V1 voxels.
        contact_pos: [B, N, 3]  ->  prf: [B, N, 3], validity: [B, N]
        """
        v1 = self.v1_pos.unsqueeze(0).expand(contact_pos.shape[0], -1, -1)
        dists = torch.cdist(contact_pos, v1)
        weights = torch.exp(-dists ** 2 / (2.0 * self.soft_match_sigma ** 2))
        validity = weights.sum(dim=-1).clamp(max=1.0)
        weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        prf = torch.einsum("bcv,vp->bcp", weights_norm, self.v1_prf)
        return prf, validity

    def _render(
        self, prf: torch.Tensor, validity: torch.Tensor, electrode_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked differentiable gaussian splatting. -> [B, 1, H, W]"""
        B, N = prf.shape[0], prf.shape[1]
        H = W = self.map_size
        device = prf.device

        angle_rad = prf[:, :, 0] * _DEG2RAD
        ecc = prf[:, :, 1]

        # Cortical magnification (wedge-dipole)
        m = _CMAG_K * (1.0 / (ecc + _CMAG_A) - 1.0 / (ecc + _CMAG_B))
        m_inv = 1.0 / (m.abs() + 1e-8)
        phos_sigma = (_SPREAD * m_inv) / 2.0

        scaled_ecc = H / VIEW_ANGLE
        cx = ecc * torch.cos(angle_rad) * scaled_ecc + H / 2.0
        cy = ecc * torch.sin(angle_rad) * scaled_ecc + W / 2.0
        phos_size = (phos_sigma * scaled_ecc).clamp(min=1.0)
        w = electrode_prob * validity

        yy = torch.arange(H, device=device, dtype=torch.float32)
        xx = torch.arange(W, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        gx = grid_x.unsqueeze(0).unsqueeze(0)
        gy = grid_y.unsqueeze(0).unsqueeze(0)

        phosphene_map = torch.zeros(B, H, W, device=device)
        chunk = self.render_chunk_size
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            cx_ = cx[:, start:end, None, None]
            cy_ = cy[:, start:end, None, None]
            s_ = phos_size[:, start:end, None, None]
            w_ = w[:, start:end, None, None]
            gaussians = torch.exp(-((gx - cx_) ** 2 + (gy - cy_) ** 2) / (s_ ** 2 + 1e-8))
            phosphene_map = phosphene_map + (gaussians * w_).sum(dim=1)

        phosphene_map = torch.rot90(phosphene_map, k=1, dims=(-2, -1))

        # Max-normalize to [0, 1] (matching dataset normalization)
        flat = phosphene_map.flatten(start_dim=1)
        max_val = flat.max(dim=1).values.view(B, 1, 1)
        phosphene_map = phosphene_map / (max_val + 1e-8)

        return phosphene_map.unsqueeze(1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, params: torch.Tensor, electrode_logits: torch.Tensor) -> torch.Tensor:
        """
        params: [B, 4]  bounded (alpha, beta, offset, shank_length)
        electrode_logits: [B, 1000]  raw logits (sigmoid applied here)
        Returns: [B, 1, H, W]
        """
        electrode_prob = torch.sigmoid(electrode_logits)
        alpha, beta, offset, shank = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

        positioned = self._create_positioned_grid(alpha, beta, offset, shank)
        prf, validity = self._soft_prf_lookup(positioned)
        return self._render(prf, validity, electrode_prob)
