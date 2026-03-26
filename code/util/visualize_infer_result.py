from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Visualize target / recon for a single PhosOpt inference run"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=project_root / "results" / "emnist_five_infer",
        help="Directory that contains <tag>.target.npy and <tag>.recon.npy "
             "(default: results/emnist_five_infer)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag used during infer.py (e.g., five_first)",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Output PNG path (default: <results-dir>/<tag>_viz.png)",
    )
    return parser.parse_args()


def _load_map(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    arr = np.load(path).astype("float32")
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected [H,W] or [1,H,W], got shape {arr.shape} for {path}")
    return arr


def normalize(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = m.astype("float32")
    max_val = float(m.max())
    if max_val > eps:
        m = m / max_val
    return m


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    tag = args.tag

    target_path = results_dir / f"{tag}.target.npy"
    recon_path = results_dir / f"{tag}.recon.npy"

    target = _load_map(target_path)
    recon = _load_map(recon_path)

    # normalize both to [0,1] for fair visualization
    target_n = normalize(target)
    recon_n = normalize(recon)
    diff = np.abs(target_n - recon_n)

    cmap = "jet"

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    ax = axes[0]
    im0 = ax.imshow(target_n, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title("Target")
    ax.axis("off")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im1 = ax.imshow(recon_n, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title("Reconstruction")
    ax.axis("off")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    im2 = ax.imshow(diff, cmap="seismic")
    ax.set_title("|Target - Recon|")
    ax.axis("off")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    if args.out_path is None:
        out_path = results_dir / f"{tag}_viz.png"
    else:
        out_path = args.out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved visualization: {out_path}")


if __name__ == "__main__":
    main()

