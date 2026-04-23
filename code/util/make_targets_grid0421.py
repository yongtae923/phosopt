from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "targets0421"
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_DIR / "targets0421_grid.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a single grid image from all .npy target maps in a folder."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing .npy target maps.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output image path (png).",
    )
    p.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of columns in the grid.",
    )
    p.add_argument(
        "--cell-size",
        type=int,
        default=128,
        help="Cell size in pixels (maps are resized to this for display).",
    )
    p.add_argument(
        "--padding",
        type=int,
        default=16,
        help="Padding between cells in pixels.",
    )
    p.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide filename labels under each map.",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="jet",
        help="Matplotlib colormap name (default: jet).",
    )
    p.add_argument(
        "--title",
        type=str,
        default="Target Maps",
        help="Figure title.",
    )
    return p.parse_args()


def _load_map(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D map in {path}, got shape={arr.shape}")

    return arr


def main() -> int:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.cols <= 0:
        raise ValueError("--cols must be > 0")
    if args.cell_size <= 0:
        raise ValueError("--cell-size must be > 0")
    if args.padding < 0:
        raise ValueError("--padding must be >= 0")

    files = sorted(args.input_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in: {args.input_dir}")

    cols = args.cols
    rows = math.ceil(len(files) / cols)
    fig_w = cols * 2.0
    fig_h = rows * 2.0
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), facecolor="#f0f0f0")
    fig.suptitle(args.title)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, path in enumerate(files):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        arr = _load_map(path)
        ax.imshow(arr, cmap=args.cmap)
        ax.axis("off")
        if not args.no_labels:
            ax.set_title(path.stem, fontsize=12)

    total_axes = rows * cols
    for i in range(len(files), total_axes):
        row = i // cols
        col = i % cols
        axes[row, col].axis("off")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"saved: {args.output}")
    print(f"count: {len(files)}")
    print(f"grid: {rows} x {cols}")
    print(f"figsize: {fig_w:.2f} x {fig_h:.2f} (inch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
