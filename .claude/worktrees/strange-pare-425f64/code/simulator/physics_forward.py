from __future__ import annotations

import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASECODE_DIR = PROJECT_ROOT / "basecode"
if str(BASECODE_DIR) not in sys.path:
    sys.path.insert(0, str(BASECODE_DIR))

from ninimplant import get_xyz
from electphos import cortical_spread, create_grid, get_cortical_magnification, implant_grid


FNAME_ANG = "inferred_angle.mgz"
FNAME_ECC = "inferred_eccen.mgz"
FNAME_SIGMA = "inferred_sigma.mgz"
FNAME_APARC = "aparc+aseg.mgz"
FNAME_LABEL = "inferred_varea.mgz"

WINDOWSIZE = 1000
N_CONTACTPOINTS_SHANK = 10
SPACING_ALONG_XY = 1
CORT_MAG_MODEL = "wedge-dipole"
VIEW_ANGLE = 90
AMP = 100
EXPECTED_ELECTRODE_COUNT = 1000


def coords_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.empty((3, 0), dtype=np.int32)
    aset = set(map(tuple, np.round(a).T.astype(np.int32)))
    bset = set(map(tuple, np.round(b).T.astype(np.int32)))
    inter = list(aset & bset)
    if not inter:
        return np.empty((3, 0), dtype=np.int32)
    return np.array(inter, dtype=np.int32).T


def normalize_phosphene_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64, copy=True)
    max_val = np.max(arr)
    if max_val > 0:
        arr /= max_val
    return arr.astype(np.float32)


def validate_electrode_activation(
    electrode_activation: np.ndarray | None,
    expected_count: int,
) -> np.ndarray:
    if electrode_activation is None:
        return np.ones(expected_count, dtype=np.float32)
    weights = np.asarray(electrode_activation, dtype=np.float32).reshape(-1)
    if weights.size != expected_count:
        raise ValueError(f"Expected {expected_count} activations, got {weights.size}")
    return np.clip(weights, 0.0, 1.0)


def build_weighted_phosphenes(
    contacts_xyz: np.ndarray,
    good_coords: np.ndarray,
    polar_map: np.ndarray,
    ecc_map: np.ndarray,
    sigma_map: np.ndarray,
    contact_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    good_lookup = {}
    good_coords_t = np.asarray(good_coords).T.astype(np.int32)
    for idx, xyz in enumerate(good_coords_t):
        good_lookup[(int(xyz[0]), int(xyz[1]), int(xyz[2]))] = idx

    contacts_rounded = np.round(contacts_xyz.T).astype(np.int32)
    phos_rows: list[list[float]] = []
    phos_weights: list[float] = []
    for contact_idx, xyz in enumerate(contacts_rounded):
        key = (int(xyz[0]), int(xyz[1]), int(xyz[2]))
        if key not in good_lookup:
            continue
        gm_idx = good_lookup[key]
        xg, yg, zg = good_coords[:, gm_idx]
        pol = float(polar_map[int(xg), int(yg), int(zg)])
        ecc = float(ecc_map[int(xg), int(yg), int(zg)])
        sigma = float(sigma_map[int(xg), int(yg), int(zg)])
        phos_rows.append([pol, ecc, sigma])
        phos_weights.append(float(contact_weights[contact_idx]))

    if not phos_rows:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(phos_rows, dtype=np.float32), np.asarray(phos_weights, dtype=np.float32)


def prf_to_phos_weighted(
    phosphene_map: np.ndarray,
    phosphenes: np.ndarray,
    weights: np.ndarray,
    view_angle: float = 90.0,
) -> np.ndarray:
    window_size = phosphene_map.shape[1]
    scaled_ecc = window_size / view_angle

    for i in range(phosphenes.shape[0]):
        w = float(weights[i])
        if w <= 0:
            continue

        s = int(phosphenes[i, 2] * scaled_ecc)
        c_x = phosphenes[i, 1] * np.cos(math.radians(phosphenes[i, 0]))
        c_y = phosphenes[i, 1] * np.sin(math.radians(phosphenes[i, 0]))
        x = int(c_x * scaled_ecc + window_size / 2)
        y = int(c_y * scaled_ecc + window_size / 2)

        if s < 2:
            s = 2
        elif (s % 2) != 0:
            s += 1

        sigma = max(s, 1)
        grid = np.arange(0, sigma * 5, 1, dtype=np.float32)
        yy = grid[:, np.newaxis]
        x0 = y0 = (sigma * 5) // 2
        g = np.exp(-((grid - x0) ** 2 + (yy - y0) ** 2) / float(sigma**2))
        g /= max(g.max(), 1e-8)
        g *= w
        half_gauss = g.shape[0] // 2

        try:
            phosphene_map[y - half_gauss : y + half_gauss, x - half_gauss : x + half_gauss] += g
        except Exception:
            pass

    phosphene_map = np.rot90(phosphene_map, 1)
    return phosphene_map


def load_retinotopy(data_dir: Path) -> dict[str, np.ndarray]:
    data_dir = Path(data_dir)
    polar_map = nib.load(str(data_dir / FNAME_ANG)).get_fdata()
    ecc_map = nib.load(str(data_dir / FNAME_ECC)).get_fdata()
    sigma_map = nib.load(str(data_dir / FNAME_SIGMA)).get_fdata()
    aparc_roi = nib.load(str(data_dir / FNAME_APARC)).get_fdata()
    label_map = nib.load(str(data_dir / FNAME_LABEL)).get_fdata()

    dot = ecc_map * polar_map
    good_coords = np.asarray(np.where(dot != 0.0))

    cs_coords_rh = np.where(aparc_roi == 1021)
    cs_coords_lh = np.where(aparc_roi == 2021)
    gm_coords_rh = np.where((aparc_roi >= 1000) & (aparc_roi < 2000))
    gm_coords_lh = np.where(aparc_roi > 2000)

    xl, yl, zl = get_xyz(gm_coords_lh)
    xr, yr, zr = get_xyz(gm_coords_rh)
    gm_lh = np.array([xl, yl, zl]).T
    gm_rh = np.array([xr, yr, zr]).T

    v1_coords = np.asarray(np.where(label_map == 1))
    v1_coords_lh = coords_intersection(v1_coords, np.asarray(gm_coords_lh))
    v1_coords_rh = coords_intersection(v1_coords, np.asarray(gm_coords_rh))

    median_lh = [np.median(cs_coords_lh[0]), np.median(cs_coords_lh[1]), np.median(cs_coords_lh[2])]
    median_rh = [np.median(cs_coords_rh[0]), np.median(cs_coords_rh[1]), np.median(cs_coords_rh[2])]

    return {
        "polar_map": polar_map,
        "ecc_map": ecc_map,
        "sigma_map": sigma_map,
        "gm_lh": gm_lh,
        "gm_rh": gm_rh,
        "median_lh": median_lh,
        "median_rh": median_rh,
        "v1_coords_lh": v1_coords_lh,
        "v1_coords_rh": v1_coords_rh,
        "good_coords": good_coords,
    }


def make_phosphene_map(
    data_dir: Path,
    alpha: float,
    beta: float,
    offset_from_base: float,
    shank_length: float,
    electrode_activation: np.ndarray | None = None,
    hemisphere: str = "LH",
) -> np.ndarray:
    data = load_retinotopy(data_dir)
    if hemisphere.upper() == "LH":
        gm_mask = data["gm_lh"]
        start_location = data["median_lh"]
        v1_h = data["v1_coords_lh"]
    else:
        gm_mask = data["gm_rh"]
        start_location = data["median_rh"]
        v1_h = data["v1_coords_rh"]

    polar_map = data["polar_map"]
    ecc_map = data["ecc_map"]
    sigma_map = data["sigma_map"]

    new_angle = (float(alpha), float(beta), 0.0)
    orig_grid = create_grid(
        start_location,
        shank_length=float(shank_length),
        n_contactpoints_shank=N_CONTACTPOINTS_SHANK,
        spacing_along_xy=SPACING_ALONG_XY,
        offset_from_origin=0,
    )
    _, contacts_xyz_moved, *_ = implant_grid(
        gm_mask, orig_grid, start_location, new_angle, float(offset_from_base)
    )

    contact_weights = validate_electrode_activation(
        electrode_activation=electrode_activation,
        expected_count=contacts_xyz_moved.shape[1],
    )
    phos_v1, phos_weights = build_weighted_phosphenes(
        contacts_xyz=contacts_xyz_moved,
        good_coords=v1_h,
        polar_map=polar_map,
        ecc_map=ecc_map,
        sigma_map=sigma_map,
        contact_weights=contact_weights,
    )
    if phos_v1.size == 0:
        return np.zeros((WINDOWSIZE, WINDOWSIZE), dtype=np.float32)

    m_inv = 1 / get_cortical_magnification(phos_v1[:, 1], CORT_MAG_MODEL)
    spread = cortical_spread(AMP)
    phos_v1[:, 2] = (spread * m_inv) / 2

    phosphene_map = np.zeros((WINDOWSIZE, WINDOWSIZE), dtype=np.float32)
    phosphene_map = prf_to_phos_weighted(
        phosphene_map=phosphene_map,
        phosphenes=phos_v1,
        weights=phos_weights,
        view_angle=VIEW_ANGLE,
    )
    return normalize_phosphene_map(phosphene_map)
