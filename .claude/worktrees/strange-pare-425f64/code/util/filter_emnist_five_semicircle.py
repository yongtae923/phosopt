from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description=(
            "Apply basecode-like semicircle (sector) mask to emnist_letters_five.npz "
            "phosphenes and save a new npz."
        )
    )
    p.add_argument(
        "--in-npz",
        type=Path,
        default=project_root / "data" / "letters" / "emnist_letters_five.npz",
        help="Input NPZ (default: data/letters/emnist_letters_five.npz)",
    )
    p.add_argument(
        "--out-npz",
        type=Path,
        default=project_root / "data" / "letters" / "emnist_letters_five_filtered.npz",
        help="Output NPZ path (default: data/letters/emnist_letters_five_filtered.npz)",
    )
    p.add_argument(
        "--angle1",
        type=float,
        default=-90.0,
        help="Sector start angle in degrees (default: -90, matches basecode)",
    )
    p.add_argument(
        "--angle2",
        type=float,
        default=90.0,
        help="Sector end angle in degrees (default: 90, matches basecode)",
    )
    p.add_argument(
        "--radius-low-frac",
        type=float,
        default=0.0,
        help="Inner radius as fraction of half-size (default: 0.0)",
    )
    p.add_argument(
        "--radius-high-frac",
        type=float,
        default=1.0,
        help="Outer radius as fraction of half-size (default: 1.0 -> full half-disk)",
    )
    p.add_argument(
        "--phos-key",
        type=str,
        default="phosphenes",
        help="Key for phosphene maps array in NPZ (default: phosphenes)",
    )
    p.add_argument(
        "--labels-key",
        type=str,
        default="labels",
        help="Key for labels array in NPZ (default: labels; optional)",
    )
    p.add_argument(
        "--letters-key",
        type=str,
        default="letters",
        help="Key for letters array in NPZ (default: letters; optional)",
    )
    p.add_argument(
        "--clip",
        action="store_true",
        help="Clip output maps to [0,1] after masking.",
    )
    p.add_argument(
        "--viz-out-path",
        type=Path,
        default=None,
        help="Optional path to save visualization PNG. "
             "If not set, saves to 'results/emnist_five_viz/all_maps_grid.png'. "
             "Set to 'none' to disable.",
    )
    return p.parse_args()


def sector_mask(
    shape: Tuple[int, int],
    centre: Tuple[float, float],
    radius_low: float,
    radius_high: float,
    angle_range: Tuple[float, float],
) -> np.ndarray:
    """
    Ported from basecode (lossfunc.py / visualsectors.py) to match orientation.

    Returns a boolean mask for a circular sector.
    The start/stop angles are in degrees and should be given in clockwise order.
    """
    x, y = np.ogrid[: shape[0], : shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    if tmax < tmin:
        tmax += 2 * np.pi

    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin
    theta %= 2 * np.pi

    circmask_low = r2 >= radius_low * radius_low
    circmask_high = r2 <= radius_high * radius_high
    circmask = circmask_low * circmask_high

    anglemask = theta <= (tmax - tmin)
    return circmask * anglemask


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    if not args.in_npz.exists():
        raise FileNotFoundError(f"Input NPZ not found: {args.in_npz}")

    with np.load(args.in_npz, allow_pickle=False) as d:
        if args.phos_key not in d.files:
            raise KeyError(f"Key '{args.phos_key}' not found. Available: {d.files}")
        phos = d[args.phos_key].astype("float32")
        labels = d[args.labels_key] if args.labels_key in d.files else None
        letters = d[args.letters_key] if args.letters_key in d.files else None

    if phos.ndim != 3:
        raise ValueError(f"Expected phosphenes shape [N,H,W], got {phos.shape}")

    n, h, w = phos.shape
    half = min(h, w) / 2.0
    r_low = float(args.radius_low_frac) * half
    r_high = float(args.radius_high_frac) * half
    if not (0.0 <= args.radius_low_frac <= args.radius_high_frac):
        raise ValueError("--radius-low-frac must be <= --radius-high-frac and both >= 0")

    centre = ((h - 1) / 2.0, (w - 1) / 2.0)
    mask = sector_mask(
        shape=(h, w),
        centre=centre,
        radius_low=r_low,
        radius_high=r_high,
        angle_range=(float(args.angle1), float(args.angle2)),
    ).astype(bool)

    masked = phos.copy()
    masked[:, ~mask] = 0.0
    if args.clip:
        masked = np.clip(masked, 0.0, 1.0)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_kwargs = {args.phos_key: masked}
    if labels is not None:
        out_kwargs[args.labels_key] = labels
    if letters is not None:
        out_kwargs[args.letters_key] = letters
    np.savez_compressed(args.out_npz, **out_kwargs)

    # ------------------------------------------------------------------
    # Visualization (grid only, basecode-like colormap)
    # ------------------------------------------------------------------
    viz_out: Path | None
    if args.viz_out_path is None:
        viz_out = project_root / "results" / "emnist_five_viz" / "all_maps_grid.png"
    else:
        if str(args.viz_out_path).lower() in {"none", "null", "false", "0"}:
            viz_out = None
        else:
            viz_out = args.viz_out_path

    if viz_out is not None:
        def _title_for(i: int) -> str:
            parts = [f"map_{i}"]
            if letters is not None:
                try:
                    parts.append(f"letter={str(np.asarray(letters)[i])}")
                except Exception:
                    pass
            if labels is not None:
                try:
                    parts.append(f"label={int(np.asarray(labels)[i])}")
                except Exception:
                    pass
            return " | ".join(parts)

        cmap = "jet"
        cols = min(5, int(masked.shape[0]))
        rows = int(np.ceil(masked.shape[0] / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        if rows == 1:
            axes = np.array([axes])
        axes = axes.reshape(rows, cols)

        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            if idx < masked.shape[0]:
                m = masked[idx]
                # Normalize for visualization (keep relative within map)
                max_val = float(m.max())
                if max_val > 1e-8:
                    m = m / max_val
                ax.imshow(m, cmap=cmap, vmin=0.0, vmax=1.0)
                ax.set_title(_title_for(idx))
                ax.axis("off")
            else:
                ax.axis("off")
        fig.tight_layout()
        viz_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(viz_out, dpi=150)
        plt.close(fig)
        print(f"[filter] saved viz: {viz_out}")

    kept_frac = float(mask.mean())
    print(f"[filter] in:  {args.in_npz}")
    print(f"[filter] out: {args.out_npz}")
    print(f"[filter] phos shape: {phos.shape} -> {masked.shape}")
    print(
        f"[filter] mask kept fraction: {kept_frac:.3f} "
        f"(angle={args.angle1:.1f}..{args.angle2:.1f}, "
        f"r_frac={args.radius_low_frac:.2f}..{args.radius_high_frac:.2f})"
    )


if __name__ == "__main__":
    main()

