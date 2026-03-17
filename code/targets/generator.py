"""
Target map generator for PhosOpt experiments.

Produces 4 categories x 5 variants = 20 target maps at a given resolution.
Categories (exp.md Section 4):
  1. single blob     -- single Gaussian at varying positions/sizes
  2. multiple blobs  -- 2-4 Gaussians combined
  3. arc shapes      -- sector / arc masks with Gaussian fill
  4. MNIST letters   -- selected from pre-generated EMNIST phosphene maps

All maps are normalised to [0, 1] and restricted to the right visual
hemi-field (matching the basecode convention of angle range [-90, 90] deg).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers (adapted from basecode/visualsectors.py)
# ---------------------------------------------------------------------------

def _make_gaussian(size: int, fwhm: float, center: tuple[int, int] | None = None) -> np.ndarray:
    x = np.arange(0, size, 1, dtype=np.float32)
    y = x[:, np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0, y0 = center
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / max(fwhm, 1e-6) ** 2)


def _sector_mask(
    shape: tuple[int, int],
    centre: tuple[int, int],
    radius_low: float,
    radius_high: float,
    angle_range: tuple[float, float],
) -> np.ndarray:
    x, y = np.ogrid[: shape[0], : shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)
    if tmax < tmin:
        tmax += 2 * np.pi
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    theta = np.arctan2(x - cx, y - cy) - tmin
    theta %= 2 * np.pi
    circ = (r2 >= radius_low ** 2) & (r2 <= radius_high ** 2)
    ang = theta <= (tmax - tmin)
    return (circ & ang).astype(np.float32)


def _normalise(m: np.ndarray) -> np.ndarray:
    mx = m.max()
    if mx > 0:
        m = m / mx
    return m.astype(np.float32)


# ---------------------------------------------------------------------------
# Category generators
# ---------------------------------------------------------------------------

def generate_single_blobs(size: int = 256, n: int = 5, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    half = size // 2
    maps: list[np.ndarray] = []
    offsets = [
        (half, half),
        (half - size // 5, half),
        (half + size // 5, half),
        (half, half - size // 5),
        (half, half + size // 5),
    ]
    fwhms = rng.integers(size // 6, size // 3, size=n)
    for i in range(n):
        g = _make_gaussian(size, float(fwhms[i]), center=offsets[i % len(offsets)])
        mask = _sector_mask((size, size), (half, half), 0, half, (-90, 90))
        g *= mask
        maps.append(_normalise(g))
    return maps


def generate_multi_blobs(size: int = 256, n: int = 5, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + 100)
    half = size // 2
    maps: list[np.ndarray] = []
    for _ in range(n):
        canvas = np.zeros((size, size), dtype=np.float32)
        n_blobs = rng.integers(2, 5)
        for _ in range(n_blobs):
            cx = int(rng.integers(half // 2, size - half // 4))
            cy = int(rng.integers(half // 4, size - half // 4))
            fwhm = float(rng.integers(size // 10, size // 4))
            canvas += _make_gaussian(size, fwhm, center=(cx, cy))
        mask = _sector_mask((size, size), (half, half), 0, half, (-90, 90))
        canvas *= mask
        maps.append(_normalise(canvas))
    return maps


def generate_arcs(size: int = 256, n: int = 5, seed: int = 0) -> list[np.ndarray]:
    """
    Arc shapes: vimplant basecode 4 (upper_sector, lower_sector, inner_ring, complete_gauss)
    plus one extra (outer_ring). Scaled from basecode WINDOWSIZE=1000, radius 0–500, fwhm=400.
    """
    half = size // 2
    # Scale basecode 1000 -> size: radius 500 -> half, 250 -> half*0.5, fwhm 400 -> size*0.4
    r_full = half
    r_inner = half * 0.5
    fwhm = size * 0.4
    centre = (half, half)
    # (radius_low, radius_high, angle1_deg, angle2_deg) — same convention as basecode sector_mask
    configs = [
        (0, r_full, -90, -45),   # upper_sector (basecode)
        (0, r_full, 45, 90),    # lower_sector (basecode)
        (0, r_inner, -90, 90),  # inner_ring (basecode)
        (0, r_full, -90, 90),   # complete_gauss (basecode)
        (r_inner, r_full, -90, 90),  # outer_ring (basecode) — 5th
    ]
    maps: list[np.ndarray] = []
    for i in range(n):
        rl, rh, a1, a2 = configs[i % len(configs)]
        g = _make_gaussian(size, fwhm, center=centre)
        mask = _sector_mask((size, size), centre, rl, rh, (a1, a2))
        canvas = g * mask
        maps.append(_normalise(canvas))
    return maps


def _apply_hemifield(m: np.ndarray) -> np.ndarray:
    """Apply right hemifield sector mask (angle -90 to 90 deg) to a map."""
    h, w = m.shape
    half = h // 2
    centre = (half, half)
    mask = _sector_mask((h, w), centre, 0, half, (-90, 90))
    return (m.astype(np.float32) * mask).astype(np.float32)


def generate_mnist_letters(
    size: int = 256,
    n: int = 5,
    seed: int = 0,
    npz_path: str | Path | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Load letter target maps from an npz; apply hemifield mask; return (maps, name_suffixes).

    - If *npz_path* points to a "five"-style npz (has key "phosphenes"): use those
      maps, apply hemifield, first *n*; names from "letters" key if present else "00","01",...
    - Else if *npz_path* points to a full dataset: sample *n* at random, apply hemifield.
    - If *npz_path* is None or file missing: fallback synthetic patterns.
    """
    if npz_path is not None:
        p = Path(npz_path)
        if p.exists():
            with np.load(p, allow_pickle=False) as f:
                if "phosphenes" in f.files:
                    # Five-style npz: use as-is then apply hemifield
                    raw = f["phosphenes"].astype(np.float32)
                    if raw.ndim == 4 and raw.shape[1] == 1:
                        raw = raw[:, 0, :, :]
                    maps = [_apply_hemifield(raw[i].copy()) for i in range(min(n, len(raw)))]
                    if "letters" in f.files:
                        letters = [str(f["letters"][i]) for i in range(min(n, len(maps)))]
                    else:
                        letters = [f"{i:02d}" for i in range(len(maps))]
                    return (maps, letters)
                # Full dataset: random pick then apply hemifield
                key = "test_phosphenes" if "test_phosphenes" in f.files else f.files[0]
                raw = f[key].astype(np.float32)
            if raw.ndim == 4 and raw.shape[1] == 1:
                raw = raw[:, 0, :, :]
            rng = np.random.default_rng(seed + 200)
            idx = rng.choice(len(raw), size=min(n, len(raw)), replace=False)
            maps = [_apply_hemifield(raw[i].copy()) for i in idx]
            return (maps, [f"{i:02d}" for i in range(len(maps))])

    # Fallback: simple synthetic letter-like patterns
    rng = np.random.default_rng(seed + 200)
    half = size // 2
    maps = []
    for _ in range(n):
        canvas = np.zeros((size, size), dtype=np.float32)
        n_strokes = rng.integers(2, 5)
        for _ in range(n_strokes):
            cx = int(rng.integers(half // 2, size - half // 4))
            cy = int(rng.integers(half // 4, size - half // 4))
            fwhm = float(rng.integers(size // 12, size // 6))
            canvas += _make_gaussian(size, fwhm, center=(cx, cy))
        mask = _sector_mask((size, size), (half, half), 0, half, (-90, 90))
        canvas *= mask
        maps.append(_normalise(canvas))
    return (maps, [f"{i:02d}" for i in range(len(maps))])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_all_targets(
    size: int = 256,
    seed: int = 0,
    npz_path: str | Path | None = None,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[str]]]:
    """Return (targets, category_suffixes). category_suffixes has per-file names for mnist_letter (e.g. T,A,E,O,X)."""
    letter_maps, letter_suffixes = generate_mnist_letters(size, 5, seed, npz_path)
    targets = {
        "single_blob": generate_single_blobs(size, 5, seed),
        "multi_blob": generate_multi_blobs(size, 5, seed),
        "arc": generate_arcs(size, 5, seed),
        "mnist_letter": letter_maps,
    }
    category_suffixes = {"mnist_letter": letter_suffixes}
    return (targets, category_suffixes)


def save_targets(
    targets: dict[str, list[np.ndarray]],
    output_dir: str | Path,
    category_suffixes: dict[str, list[str]] | None = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for category, maps in targets.items():
        suffixes = None
        if category_suffixes and category in category_suffixes:
            suffixes = category_suffixes[category]
        for i, m in enumerate(maps):
            suffix = (suffixes[i] if suffixes and i < len(suffixes) else f"{i:02d}")
            np.save(out / f"{category}_{suffix}.npy", m)
    print(f"Saved {sum(len(v) for v in targets.values())} target maps to {out}")


def load_targets(target_dir: str | Path) -> dict[str, np.ndarray]:
    """Load all .npy target maps from a directory, keyed by stem name."""
    d = Path(target_dir)
    result: dict[str, np.ndarray] = {}
    for p in sorted(d.glob("*.npy")):
        result[p.stem] = np.load(p).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment target maps")
    parser.add_argument("--output", type=Path, default=Path("data/targets"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("data/letters/emnist_letters_five.npz"),
        help="Letter targets npz: use data/letters/emnist_letters_five.npz (phosphenes as-is) or full EMNIST npz (random pick)",
    )
    args = parser.parse_args()

    targets, category_suffixes = generate_all_targets(size=args.resolution, seed=args.seed, npz_path=args.npz)
    save_targets(targets, args.output, category_suffixes=category_suffixes)

    for cat, maps in targets.items():
        print(f"  {cat}: {len(maps)} maps, shape {maps[0].shape}, "
              f"range [{maps[0].min():.3f}, {maps[0].max():.3f}]")


if __name__ == "__main__":
    main()
