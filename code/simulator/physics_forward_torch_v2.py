# D:\yongtae\phosopt\code\simulator\physics_forward_torch_v2.py

"""
Differentiable phosphene simulator

The main goal is to have a fully differentiable simulator that does not rely on 
    any non-PyTorch code or precomputation, so that it can be used for 
    gradient-based inverse learning.

Key features:
- Loads retinotopy/anatomy files directly
- Pre-computes a surface-distance LUT using trimesh for fast ray-surface 
    intersection during forward
- Uses soft Gaussian matching from contact positions to V1 voxel pRFs for 
    differentiable rendering
- Implements chunked Gaussian splatting for efficient rendering of phosphenes
"""

from __future__ import annotations

import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Inlined constants
# ---------------------------------------------------------------------

FNAME_ANG = "inferred_angle.mgz"
FNAME_ECC = "inferred_eccen.mgz"
FNAME_SIGMA = "inferred_sigma.mgz"
FNAME_APARC = "aparc+aseg.mgz"
FNAME_LABEL = "inferred_varea.mgz"

WINDOWSIZE = 1000
N_CONTACTPOINTS_SHANK = 10
SPACING_ALONG_XY = 1.0
CORT_MAG_MODEL = "wedge-dipole"
VIEW_ANGLE = 90.0
AMP = 100.0
EXPECTED_ELECTRODE_COUNT = 1000

_DEG2RAD = math.pi / 180.0
_SPREAD = math.sqrt(AMP / 675.0)
_CMAG_A = 0.75
_CMAG_B = 120.0
_CMAG_K = 17.3


# ---------------------------------------------------------------------
# Retinotopy loading helpers
# ---------------------------------------------------------------------

def _coords_from_where(where_tuple: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Convert np.where output to [N, 3] array.
    """
    if len(where_tuple) != 3:
        raise ValueError("Expected a 3-tuple from np.where for 3D volume coordinates.")
    x, y, z = where_tuple
    if x.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def coords_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Intersect two coordinate sets stored as [3, N] integer arrays.
    Returns [3, K].
    """
    if a.size == 0 or b.size == 0:
        return np.empty((3, 0), dtype=np.int32)

    aset = set(map(tuple, np.round(a).T.astype(np.int32)))
    bset = set(map(tuple, np.round(b).T.astype(np.int32)))
    inter = list(aset & bset)

    if not inter:
        return np.empty((3, 0), dtype=np.int32)

    return np.array(inter, dtype=np.int32).T


def load_retinotopy(data_dir: str | Path) -> dict[str, np.ndarray]:
    """
    Load subject-specific retinotopy / anatomy files.
    Returns a dict compatible with both the exact and differentiable simulators.
    """
    data_dir = Path(data_dir)

    polar_map = nib.load(str(data_dir / FNAME_ANG)).get_fdata()
    ecc_map = nib.load(str(data_dir / FNAME_ECC)).get_fdata()
    sigma_map = nib.load(str(data_dir / FNAME_SIGMA)).get_fdata()
    aparc_roi = nib.load(str(data_dir / FNAME_APARC)).get_fdata()
    label_map = nib.load(str(data_dir / FNAME_LABEL)).get_fdata()

    dot = ecc_map * polar_map
    good_coords = np.asarray(np.where(dot != 0.0), dtype=np.int32)

    # Calcarine sulcus labels from aparc+aseg
    cs_coords_rh = np.where(aparc_roi == 1021)
    cs_coords_lh = np.where(aparc_roi == 2021)

    # Gray matter masks
    gm_coords_rh = np.where((aparc_roi >= 1000) & (aparc_roi < 2000))
    gm_coords_lh = np.where(aparc_roi > 2000)

    gm_rh = _coords_from_where(gm_coords_rh)
    gm_lh = _coords_from_where(gm_coords_lh)

    # V1 label == 1, then intersect with hemisphere-specific GM
    v1_coords = np.asarray(np.where(label_map == 1), dtype=np.int32)
    v1_coords_lh = coords_intersection(v1_coords, np.asarray(gm_coords_lh, dtype=np.int32))
    v1_coords_rh = coords_intersection(v1_coords, np.asarray(gm_coords_rh, dtype=np.int32))

    if len(cs_coords_lh[0]) == 0 or len(cs_coords_rh[0]) == 0:
        raise ValueError("Could not find calcarine sulcus coordinates in aparc+aseg.mgz")

    median_lh = np.array(
        [
            np.median(cs_coords_lh[0]),
            np.median(cs_coords_lh[1]),
            np.median(cs_coords_lh[2]),
        ],
        dtype=np.float32,
    )
    median_rh = np.array(
        [
            np.median(cs_coords_rh[0]),
            np.median(cs_coords_rh[1]),
            np.median(cs_coords_rh[2]),
        ],
        dtype=np.float32,
    )

    return {
        "polar_map": polar_map.astype(np.float32),
        "ecc_map": ecc_map.astype(np.float32),
        "sigma_map": sigma_map.astype(np.float32),
        "gm_lh": gm_lh.astype(np.float32),
        "gm_rh": gm_rh.astype(np.float32),
        "median_lh": median_lh.astype(np.float32),
        "median_rh": median_rh.astype(np.float32),
        "v1_coords_lh": v1_coords_lh.astype(np.int32),
        "v1_coords_rh": v1_coords_rh.astype(np.int32),
        "good_coords": good_coords.astype(np.int32),
    }


def _extract_v1_prf(data: dict, hemisphere: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract V1 voxel positions and their pRF properties (angle, ecc, sigma).
    Returns:
      positions: [Nv, 3]
      prf_props: [Nv, 3] = [polar_angle, eccentricity, sigma]
    """
    hemi = hemisphere.upper()
    if hemi not in {"LH", "RH"}:
        raise ValueError("hemisphere must be 'LH' or 'RH'")

    v1_coords = data["v1_coords_lh"] if hemi == "LH" else data["v1_coords_rh"]
    polar_map = data["polar_map"]
    ecc_map = data["ecc_map"]
    sigma_map = data["sigma_map"]

    positions: list[list[float]] = []
    prf_props: list[list[float]] = []

    for i in range(v1_coords.shape[1]):
        x = int(v1_coords[0, i])
        y = int(v1_coords[1, i])
        z = int(v1_coords[2, i])

        pol = float(polar_map[x, y, z])
        ecc = float(ecc_map[x, y, z])
        sig = float(sigma_map[x, y, z])

        # Keep only meaningful retinotopy entries
        if ecc > 0 and pol != 0:
            positions.append([x, y, z])
            prf_props.append([pol, ecc, sig])

    if len(positions) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(prf_props, dtype=np.float32),
    )


# ---------------------------------------------------------------------
# Surface-distance LUT precomputation
# ---------------------------------------------------------------------

def _precompute_surface_distances(
    gm_points: np.ndarray,
    start_location: np.ndarray | list[float],
    alpha_range: tuple[float, float] = (-90.0, 90.0),
    beta_range: tuple[float, float] = (-15.0, 110.0),
    n_alpha: int = 37,
    n_beta: int = 26,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-compute ray-to-brain-surface distances over a grid of (alpha, beta).

    This uses trimesh convex hull intersection once during initialization.
    During forward, differentiable bilinear interpolation into the LUT is used.
    """
    import trimesh

    gm_points = np.asarray(gm_points, dtype=np.float64)
    if gm_points.ndim != 2 or gm_points.shape[1] != 3:
        raise ValueError("gm_points must have shape [N, 3]")

    if gm_points.shape[0] == 0:
        raise ValueError("gm_points is empty")

    mesh = trimesh.points.PointCloud(gm_points).convex_hull
    start = np.asarray(start_location, dtype=np.float64).reshape(3)

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha, dtype=np.float64)
    betas = np.linspace(beta_range[0], beta_range[1], n_beta, dtype=np.float64)
    distances = np.full((n_alpha, n_beta), 20.0, dtype=np.float32)

    for i, a_deg in enumerate(alphas):
        for j, b_deg in enumerate(betas):
            a = math.radians(float(a_deg))
            b = math.radians(float(b_deg))

            # Direction = Rx(alpha) @ Ry(beta) @ [0, 0, -1]
            d = np.array(
                [
                    -math.sin(b),
                    math.sin(a) * math.cos(b),
                    -math.cos(a) * math.cos(b),
                ],
                dtype=np.float64,
            ).reshape(1, 3)

            try:
                locs, _, _ = mesh.ray.intersects_location(
                    ray_origins=start.reshape(1, 3),
                    ray_directions=d,
                )
                if len(locs) > 0:
                    distances[i, j] = float(np.linalg.norm(locs[0] - start))
            except Exception:
                # Keep fallback distance if trimesh intersection fails
                pass

    return (
        distances.astype(np.float32),
        alphas.astype(np.float32),
        betas.astype(np.float32),
    )


# ---------------------------------------------------------------------
# Differentiable simulator
# ---------------------------------------------------------------------

class DifferentiableSimulatorIndependent(nn.Module):
    """
    Differentiable phosphene simulator for gradient-based inverse learning.

    Independent version:
      - does not import physics_forward.py
      - directly loads retinotopy/anatomy files
      - keeps same external interface as the original torch simulator

    Inputs:
      params: [B, 4] = (alpha, beta, offset_from_base, shank_length)
      electrode_logits: [B, 1000]

    Output:
      [B, 1, H, W]
    """

    def __init__(
        self,
        data_dir: str | Path,
        hemisphere: str = "LH",
        map_size: int = 256,
        soft_match_sigma: float = 1.5,
        render_chunk_size: int = 50,
        alpha_range: tuple[float, float] = (-90.0, 90.0),
        beta_range: tuple[float, float] = (-15.0, 110.0),
        lut_n_alpha: int = 37,
        lut_n_beta: int = 26,
    ) -> None:
        super().__init__()

        self.hemisphere = hemisphere.upper()
        if self.hemisphere not in {"LH", "RH"}:
            raise ValueError("hemisphere must be 'LH' or 'RH'")

        self.map_size = int(map_size)
        self.soft_match_sigma = float(soft_match_sigma)
        self.render_chunk_size = int(render_chunk_size)
        self.is_differentiable = True

        n = N_CONTACTPOINTS_SHANK
        spacing = float(SPACING_ALONG_XY)

        print(f"[DifferentiableSimulatorIndependent] Loading retinotopy from {data_dir} ...")
        data = load_retinotopy(data_dir)

        # --------------------------------------------------------------
        # V1 voxel positions and pRF properties
        # --------------------------------------------------------------
        v1_pos, v1_prf = _extract_v1_prf(data, self.hemisphere)
        if v1_pos.shape[0] == 0:
            raise ValueError("No valid V1 voxels found for this hemisphere/subject.")

        self.register_buffer("v1_pos", torch.from_numpy(v1_pos))
        self.register_buffer("v1_prf", torch.from_numpy(v1_prf))
        print(f"  V1 voxels: {v1_pos.shape[0]}")

        # --------------------------------------------------------------
        # Start location
        # --------------------------------------------------------------
        start = data["median_lh"] if self.hemisphere == "LH" else data["median_rh"]
        self.register_buffer("start_loc", torch.tensor(start, dtype=torch.float32))

        # --------------------------------------------------------------
        # Surface distance LUT
        # --------------------------------------------------------------
        gm = data["gm_lh"] if self.hemisphere == "LH" else data["gm_rh"]
        print("  Pre-computing surface distance lookup table ...")
        surf_dist, alphas, betas = _precompute_surface_distances(
            gm_points=gm,
            start_location=start,
            alpha_range=alpha_range,
            beta_range=beta_range,
            n_alpha=lut_n_alpha,
            n_beta=lut_n_beta,
        )
        self.register_buffer("surf_dist_lut", torch.from_numpy(surf_dist))
        self.register_buffer("alpha_grid", torch.from_numpy(alphas))
        self.register_buffer("beta_grid", torch.from_numpy(betas))

        # --------------------------------------------------------------
        # Base grid template
        # ordering: y-slow, x-mid, z-fast
        # shape: [1000, 3] when n=10
        # --------------------------------------------------------------
        y_vals = torch.arange(n, dtype=torch.float32) * spacing
        x_vals = torch.arange(n, dtype=torch.float32) * spacing
        z_fracs = torch.linspace(0.0, 1.0, n, dtype=torch.float32)
        yy, xx, zz = torch.meshgrid(y_vals, x_vals, z_fracs, indexing="ij")
        template = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        if template.shape[0] != EXPECTED_ELECTRODE_COUNT:
            raise ValueError(
                f"Grid template produced {template.shape[0]} contacts, "
                f"expected {EXPECTED_ELECTRODE_COUNT}"
            )

        self.register_buffer("grid_template", template)

        print("  Initialization complete.")

    # -----------------------------------------------------------------
    # Building blocks
    # -----------------------------------------------------------------

    def _rotation_matrix(self, alpha_deg: torch.Tensor, beta_deg: torch.Tensor) -> torch.Tensor:
        """
        Build Rx(alpha) @ Ry(beta).
        Input: [B], [B]
        Output: [B, 3, 3]
        """
        a = alpha_deg * _DEG2RAD
        b = beta_deg * _DEG2RAD

        ca, sa = torch.cos(a), torch.sin(a)
        cb, sb = torch.cos(b), torch.sin(b)

        ones = torch.ones_like(a)
        zeros = torch.zeros_like(a)

        rx = torch.stack(
            [
                torch.stack([ones, zeros, zeros], dim=-1),
                torch.stack([zeros, ca, -sa], dim=-1),
                torch.stack([zeros, sa, ca], dim=-1),
            ],
            dim=-2,
        )

        ry = torch.stack(
            [
                torch.stack([cb, zeros, sb], dim=-1),
                torch.stack([zeros, ones, zeros], dim=-1),
                torch.stack([-sb, zeros, cb], dim=-1),
            ],
            dim=-2,
        )

        return rx @ ry

    def _interpolate_surface_dist(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Bilinear interpolation into the pre-computed surface-distance LUT.
        Input: alpha [B], beta [B]
        Output: [B]
        """
        bsz = alpha.shape[0]

        a_min, a_max = self.alpha_grid[0], self.alpha_grid[-1]
        b_min, b_max = self.beta_grid[0], self.beta_grid[-1]

        a_norm = 2.0 * (alpha - a_min) / (a_max - a_min + 1e-8) - 1.0
        b_norm = 2.0 * (beta - b_min) / (b_max - b_min + 1e-8) - 1.0

        # grid_sample expects grid[..., 0] = x = beta-axis, grid[..., 1] = y = alpha-axis
        grid_coords = torch.stack([b_norm, a_norm], dim=-1).view(bsz, 1, 1, 2)
        lut = self.surf_dist_lut.unsqueeze(0).unsqueeze(0).expand(bsz, -1, -1, -1)

        sampled = F.grid_sample(
            lut,
            grid_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.view(bsz).clamp(min=1.0)

    def _create_positioned_grid(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        offset: torch.Tensor,
        shank: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create rotated and positioned contact grid.
        Inputs: [B] x 4
        Output: [B, 1000, 3]
        """
        bsz = alpha.shape[0]

        template = self.grid_template.unsqueeze(0).expand(bsz, -1, -1)
        xy = template[:, :, :2]
        z_scaled = template[:, :, 2:3] * shank.view(bsz, 1, 1)
        grid = torch.cat([xy, z_scaled], dim=-1)

        center = grid.mean(dim=1, keepdim=True)
        grid = grid - center

        r = self._rotation_matrix(alpha, beta)
        rotated = torch.bmm(grid, r.transpose(1, 2))

        ref = torch.tensor([0.0, 0.0, -1.0], device=alpha.device, dtype=alpha.dtype)
        direction = torch.matmul(r, ref.view(1, 3, 1)).squeeze(-1)
        direction = F.normalize(direction, dim=-1)

        surf_dist = self._interpolate_surface_dist(alpha, beta)
        penetration = surf_dist - shank / 2.0 - offset
        grid_center = self.start_loc.unsqueeze(0) + direction * penetration.unsqueeze(-1)

        return rotated + grid_center.unsqueeze(1)

    def _soft_prf_lookup(self, contact_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Soft Gaussian matching from contact positions to V1 voxel pRFs.

        contact_pos: [B, N, 3]
        returns:
          prf: [B, N, 3]  = [polar_angle, eccentricity, sigma]
          validity: [B, N]
        """
        v1 = self.v1_pos.unsqueeze(0).expand(contact_pos.shape[0], -1, -1)  # [B, Nv, 3]
        dists = torch.cdist(contact_pos, v1)  # [B, N, Nv]

        weights = torch.exp(-dists**2 / (2.0 * self.soft_match_sigma**2))
        validity = weights.sum(dim=-1).clamp(max=1.0)

        weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        prf = torch.einsum("bnv,vp->bnp", weights_norm, self.v1_prf)

        return prf, validity

    def _render(
        self,
        prf: torch.Tensor,
        validity: torch.Tensor,
        electrode_prob: torch.Tensor,
    ) -> torch.Tensor:
        """
        Chunked differentiable Gaussian splatting.

        prf: [B, N, 3] = [angle, ecc, sigma]
        validity: [B, N]
        electrode_prob: [B, N]
        returns: [B, 1, H, W]
        """
        bsz, n_contacts = prf.shape[0], prf.shape[1]
        h = w = self.map_size
        device = prf.device

        angle_rad = prf[:, :, 0] * _DEG2RAD
        ecc = prf[:, :, 1]

        # Match the original logic: phosphene size from cortical magnification
        # rather than directly using sigma_map as final rendered sigma
        m = _CMAG_K * (1.0 / (ecc + _CMAG_A) - 1.0 / (ecc + _CMAG_B))
        m_inv = 1.0 / (m.abs() + 1e-8)
        phos_sigma = (_SPREAD * m_inv) / 2.0

        scaled_ecc = h / VIEW_ANGLE
        cx = ecc * torch.cos(angle_rad) * scaled_ecc + h / 2.0
        cy = ecc * torch.sin(angle_rad) * scaled_ecc + w / 2.0

        phos_size = (phos_sigma * scaled_ecc).clamp(min=1.0)
        weight = electrode_prob * validity

        yy = torch.arange(h, device=device, dtype=torch.float32)
        xx = torch.arange(w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

        gx = grid_x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        gy = grid_y.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        phosphene_map = torch.zeros(bsz, h, w, device=device, dtype=torch.float32)

        chunk = self.render_chunk_size
        for start in range(0, n_contacts, chunk):
            end = min(start + chunk, n_contacts)

            cx_ = cx[:, start:end, None, None]
            cy_ = cy[:, start:end, None, None]
            s_ = phos_size[:, start:end, None, None]
            w_ = weight[:, start:end, None, None]

            gaussians = torch.exp(
                -((gx - cx_) ** 2 + (gy - cy_) ** 2) / (s_**2 + 1e-8)
            )
            phosphene_map = phosphene_map + (gaussians * w_).sum(dim=1)

        phosphene_map = torch.rot90(phosphene_map, k=1, dims=(-2, -1))

        flat = phosphene_map.flatten(start_dim=1)
        max_val = flat.max(dim=1).values.view(bsz, 1, 1)
        phosphene_map = phosphene_map / (max_val + 1e-8)

        return phosphene_map.unsqueeze(1)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(self, params: torch.Tensor, electrode_logits: torch.Tensor) -> torch.Tensor:
        """
        params: [B, 4]  bounded (alpha, beta, offset_from_base, shank_length)
        electrode_logits: [B, 1000] raw logits
        returns: [B, 1, H, W]
        """
        if params.ndim != 2 or params.shape[1] != 4:
            raise ValueError(f"params must have shape [B,4], got {tuple(params.shape)}")

        if electrode_logits.ndim != 2 or electrode_logits.shape[1] != EXPECTED_ELECTRODE_COUNT:
            raise ValueError(
                f"electrode_logits must have shape [B,{EXPECTED_ELECTRODE_COUNT}], "
                f"got {tuple(electrode_logits.shape)}"
            )

        electrode_prob = torch.sigmoid(electrode_logits)

        alpha = params[:, 0]
        beta = params[:, 1]
        offset = params[:, 2]
        shank = params[:, 3]

        positioned = self._create_positioned_grid(alpha, beta, offset, shank)
        prf, validity = self._soft_prf_lookup(positioned)
        return self._render(prf, validity, electrode_prob)


# ---------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------

DifferentiableSimulator = DifferentiableSimulatorIndependent