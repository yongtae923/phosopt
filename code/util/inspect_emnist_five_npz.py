from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    # util 스크립트는 code/util/ 아래에 있으므로,
    # 프로젝트 루트는 parents[2] (…/phosopt)
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Inspect internal structure of emnist_letters_five.npz"
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=project_root / "data" / "letters" / "emnist_letters_five.npz",
        help="Path to NPZ file (default: data/letters/emnist_letters_five.npz)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = args.npz_path

    print(f"[inspect] NPZ path: {npz_path}")
    if not npz_path.exists():
        print(f"[inspect] ERROR: file does not exist: {npz_path}")
        return

    with np.load(npz_path, allow_pickle=False) as data:
        keys = list(data.files)
        print(f"[inspect] keys: {keys}")
        if not keys:
            print("[inspect] (no arrays found)")
            return

        for key in keys:
            arr = data[key]
            print(f"\n[key] {key}")
            print(f"  shape : {arr.shape}")
            print(f"  dtype : {arr.dtype}")
            print(f"  size  : {arr.size}")
            print(f"  min   : {arr.min()}")
            print(f"  max   : {arr.max()}")
            print(f"  mean  : {arr.mean()}")


if __name__ == "__main__":
    main()

