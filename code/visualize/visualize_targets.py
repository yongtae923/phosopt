"""
Visualize all target maps from data/targets directory.

Loads all .npy files and displays them in a grid grouped by category.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_targets(target_dir: Path) -> dict[str, np.ndarray]:
    """Load all .npy target maps, keyed by stem name."""
    result: dict[str, np.ndarray] = {}
    for p in sorted(target_dir.glob("*.npy")):
        result[p.stem] = np.load(p).astype(np.float32)
    return result


def group_by_category(targets: dict[str, np.ndarray]) -> dict[str, list[tuple[str, np.ndarray]]]:
    """Group targets by category prefix (e.g. arc, single_blob)."""
    groups: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name, arr in targets.items():
        # category is everything before the last _XX (e.g. arc_00 -> arc)
        parts = name.rsplit("_", 1)
        cat = parts[0] if len(parts) == 2 else name
        if cat not in groups:
            groups[cat] = []
        groups[cat].append((name, arr))
    for cat in groups:
        groups[cat].sort(key=lambda x: x[0])
    return groups


def visualize_targets(
    target_dir: str | Path,
    output_dir: str | Path,
    output_name: str = "targets_grid.png",
    cmap: str = "jet",
    dpi: int = 150,
    show: bool = False,
) -> None:
    """Plot all target maps in a grid, grouped by category."""
    target_dir = Path(target_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    targets = load_targets(target_dir)
    if not targets:
        raise FileNotFoundError(f"No .npy files found in {target_dir}")

    groups = group_by_category(targets)
    categories = sorted(groups.keys())
    n_cats = len(categories)
    n_cols = max(len(groups[c]) for c in categories)
    n_rows = n_cats

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), squeeze=False)
    fig.suptitle("Target Maps", fontsize=14)

    for i, cat in enumerate(categories):
        items = groups[cat]
        for j, (name, arr) in enumerate(items):
            ax = axes[i, j]
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(name, fontsize=9)
            ax.axis("off")
        # Hide unused subplots
        for j in range(len(items), n_cols):
            axes[i, j].axis("off")

    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Visualize target maps from data/targets")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=root / "data" / "targets",
        help="Directory containing .npy target maps",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=root / "data" / "targets",
        help="Directory to save figure (default: data/targets)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="targets_grid.png",
        help="Output filename (default: targets_grid.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot in addition to saving",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="jet",
        help="Colormap name (default: jet)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figure",
    )
    args = parser.parse_args()

    visualize_targets(
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
        cmap=args.cmap,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
