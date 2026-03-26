from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    # util 스크립트는 code/util/ 아래에 있으므로,
    # 프로젝트 루트는 parents[2] (…/phosopt) 가 되어야 한다.
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Visualize phosphene maps from emnist_letters_five.npz"
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=project_root / "data" / "letters" / "emnist_letters_five.npz",
        help="Path to NPZ with 5 phosphene maps (default: data/letters/emnist_letters_five.npz)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Array key inside NPZ (default: prefer 'phosphenes' if present, else auto-detect)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "results" / "emnist_five_viz",
        help="Output directory for saved PNGs (default: results/emnist_five_viz)",
    )
    return parser.parse_args()


def load_npz(npz_path: Path, key: str | None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        keys = list(data.files)
        if not keys:
            raise ValueError(f"No arrays found in {npz_path}")
        if key is None:
            key = "phosphenes" if "phosphenes" in keys else None
            if key is None:
                if len(keys) != 1:
                    raise ValueError(
                        f"NPZ has multiple arrays {keys}; please specify --key explicitly."
                    )
                key = keys[0]
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {npz_path}. Available: {keys}")
        arr = data[key].astype("float32")
        labels = data["labels"] if "labels" in keys else None
        letters = data["letters"] if "letters" in keys else None

    # Expect shapes like [5,H,W] or [5,1,H,W]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected [N,H,W] or [N,1,H,W], got shape {arr.shape}")
    if arr.shape[0] != 5:
        print(f"[warn] expected 5 maps, but got {arr.shape[0]}; continuing anyway.")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] != arr.shape[0]:
            print(f"[warn] labels length {labels.shape[0]} != maps {arr.shape[0]}; ignoring labels.")
            labels = None
    if letters is not None:
        letters = np.asarray(letters)
        if letters.shape[0] != arr.shape[0]:
            print(f"[warn] letters length {letters.shape[0]} != maps {arr.shape[0]}; ignoring letters.")
            letters = None
    return arr, labels, letters, key


def normalize(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = m.astype("float32")
    max_val = float(m.max())
    if max_val > eps:
        m = m / max_val
    return m


def _title_for(i: int, labels: np.ndarray | None, letters: np.ndarray | None) -> str:
    parts = [f"map_{i}"]
    if letters is not None:
        try:
            parts.append(f"letter={str(letters[i])}")
        except Exception:
            pass
    if labels is not None:
        try:
            parts.append(f"label={int(labels[i])}")
        except Exception:
            pass
    return " | ".join(parts)


def save_figures(maps: np.ndarray, out_dir: Path, labels: np.ndarray | None, letters: np.ndarray | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = maps.shape[0]
    # Basecode 스타일에 맞춰 colormap은 'jet' 사용
    cmap = "jet"

    # 한 장짜리 그리드도 저장
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        if idx < n:
            m = normalize(maps[idx])
            ax.imshow(m, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_title(_title_for(idx, labels=labels, letters=letters))
            ax.axis("off")
        else:
            ax.axis("off")
    fig.tight_layout()
    grid_path = out_dir / "all_maps_grid.png"
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)
    print(f"Saved grid: {grid_path}")


def main() -> None:
    args = parse_args()
    maps, labels, letters, used_key = load_npz(args.npz_path, args.key)
    print(f"Loaded maps from {args.npz_path} (key='{used_key}'), shape={maps.shape}")
    if labels is not None:
        print(f"Loaded labels: shape={labels.shape}, dtype={labels.dtype}")
    if letters is not None:
        print(f"Loaded letters: shape={letters.shape}, dtype={letters.dtype}")
    save_figures(maps, args.out_dir, labels=labels, letters=letters)


if __name__ == "__main__":
    main()

