"""
Visualize first 5 samples from emnist_letters_phosphenes.npz and emnist_letters_upright.npz.
Optionally visualize emnist_letters_TAEOX.npz (one sample per letter T,A,E,O,X).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_first_n(npz_path: Path, n: int = 5) -> list[np.ndarray]:
    """Load first n samples from an npz. Uses test_phosphenes or first key."""
    with np.load(npz_path, allow_pickle=False) as f:
        key = "test_phosphenes" if "test_phosphenes" in f.files else f.files[0]
        raw = f[key].astype(np.float32)
    if raw.ndim == 4 and raw.shape[1] == 1:
        raw = raw[:, 0, :, :]
    return [raw[i] for i in range(min(n, len(raw)))]


def load_taeox(npz_path: Path) -> tuple[list[np.ndarray], list[str]]:
    """Load TAEOX npz with keys phosphenes and letters. Returns (maps, letter_labels)."""
    with np.load(npz_path, allow_pickle=False) as f:
        maps = f["phosphenes"].astype(np.float32)
        if maps.ndim == 4 and maps.shape[1] == 1:
            maps = maps[:, 0, :, :]
        letters = list(f["letters"]) if "letters" in f.files else [str(i) for i in range(len(maps))]
    return [maps[i] for i in range(len(maps))], [str(l) for l in letters]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    default_phosphenes = root / "data" / "letters" / "emnist_letters_phosphenes.npz"
    default_upright = root / "data" / "letters" / "emnist_letters_upright.npz"

    parser = argparse.ArgumentParser(description="Show first 5 samples from letter npz files")
    parser.add_argument("--phosphenes", type=Path, default=default_phosphenes)
    parser.add_argument("--upright", type=Path, default=default_upright)
    default_taeox = root / "data" / "letters" / "emnist_letters_TAEOX.npz"
    parser.add_argument(
        "--taeox",
        type=Path,
        default=default_taeox,
        help="TAEOX npz to visualize (default: data/letters/emnist_letters_TAEOX.npz)",
    )
    parser.add_argument("--n", type=int, default=5, help="Number of samples per file")
    parser.add_argument("--output-dir", "-o", type=Path, default=root / "data" / "targets")
    parser.add_argument("--output-name", default="letters_npz_preview.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.phosphenes.exists():
        raise FileNotFoundError(f"Not found: {args.phosphenes}")
    if not args.upright.exists():
        raise FileNotFoundError(f"Not found: {args.upright}")

    phosphenes = load_first_n(args.phosphenes, args.n)
    upright = load_first_n(args.upright, args.n)
    n_cols = max(len(phosphenes), len(upright))

    fig, axes = plt.subplots(2, n_cols, figsize=(2 * n_cols, 4), squeeze=False)
    fig.suptitle("First 5 from letter npz (phosphenes / upright)")

    for j, arr in enumerate(phosphenes):
        axes[0, j].imshow(arr, cmap="gray")
        axes[0, j].set_title(f"phosphenes #{j}")
        axes[0, j].axis("off")
    for j in range(len(phosphenes), n_cols):
        axes[0, j].axis("off")

    for j, arr in enumerate(upright):
        axes[1, j].imshow(arr, cmap="gray")
        axes[1, j].set_title(f"upright #{j}")
        axes[1, j].axis("off")
    for j in range(len(upright), n_cols):
        axes[1, j].axis("off")

    plt.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close()

    # TAEOX npz: one row with letter labels
    if args.taeox.exists():
        taeox_maps, taeox_letters = load_taeox(args.taeox)
        n_taeox = len(taeox_maps)
        fig2, ax2 = plt.subplots(1, n_taeox, figsize=(2 * n_taeox, 2.5), squeeze=False)
        fig2.suptitle("TAEOX letter phosphenes")
        for j, (arr, label) in enumerate(zip(taeox_maps, taeox_letters)):
            ax2[0, j].imshow(arr, cmap="gray")
            ax2[0, j].set_title(label)
            ax2[0, j].axis("off")
        plt.tight_layout()
        taeox_path = out_dir / "letters_TAEOX_preview.png"
        plt.savefig(taeox_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {taeox_path}")
        if args.show:
            plt.show()
        else:
            plt.close()
    else:
        print(f"TAEOX npz not found (skipped): {args.taeox}")


if __name__ == "__main__":
    main()
