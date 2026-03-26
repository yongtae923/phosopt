from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def sector_mask(
    shape: tuple[int, int],
    centre: tuple[float, float],
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

    circmask_low = r2 >= radius_low ** 2
    circmask_high = r2 <= radius_high ** 2
    anglemask = theta <= (tmax - tmin)
    return (circmask_low & circmask_high & anglemask).astype(bool)


def _extract_first_and_filter(
    arr: np.ndarray,
    index: int,
    mask: np.ndarray,
) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[1] == 1:
        sample = arr[index : index + 1, 0, :, :]
    elif arr.ndim == 3:
        sample = arr[index : index + 1, :, :]
    else:
        raise ValueError(f"Unsupported array shape for phosphene set: {arr.shape}")

    h, w = sample.shape[-2], sample.shape[-1]
    if mask.shape != (h, w):
        raise ValueError(f"Mask shape {mask.shape} does not match sample shape {(h,w)}")

    filtered = sample.copy()
    filtered[..., ~mask] = 0.0
    return filtered


def main() -> None:
    p = argparse.ArgumentParser(description="Load first sample from EMNIST phosphene npz, apply sector mask, save as small npz.")
    p.add_argument(
        "--in-npz",
        type=Path,
        required=False,
        default=Path("data/letters/emnist_letters_phosphenes.npz"),
        help="Input source npz",
    )
    p.add_argument(
        "--out-npz",
        type=Path,
        required=False,
        default=Path("data/letters/emnist_letters_phosphenes_first_filtered.npz"),
        help="Output filtered npz",
    )
    p.add_argument("--index", type=int, default=0, help="Index of sample to extract (default: 0)")
    p.add_argument("--angle1", type=float, default=-90.0)
    p.add_argument("--angle2", type=float, default=90.0)
    p.add_argument("--radius-low-frac", type=float, default=0.0)
    p.add_argument("--radius-high-frac", type=float, default=1.0)
    p.add_argument("--map-size", type=int, default=256)
    p.add_argument("--compress", action="store_true", help="Compress npz")
    args = p.parse_args()

    if not args.in_npz.exists():
        raise FileNotFoundError(f"Input NPZ not found: {args.in_npz}")

    print(f"[step] loading npz: {args.in_npz}")
    with np.load(args.in_npz, allow_pickle=False) as data:
        print(f"[step] keys in npz: {data.files}")
        for key in ["train_phosphenes", "test_phosphenes"]:
            if key not in data.files:
                raise KeyError(f"Key '{key}' not found in input npz. Available: {data.files}")

        train_arr = data["train_phosphenes"]
        test_arr = data["test_phosphenes"]
        train_labels = data.get("train_labels", None)
        test_labels = data.get("test_labels", None)

    print(f"[step] loaded train_phosphenes shape={train_arr.shape} dtype={train_arr.dtype}")
    print(f"[step] loaded test_phosphenes shape={test_arr.shape} dtype={test_arr.dtype}")
    if train_labels is not None:
        print(f"[step] loaded train_labels shape={train_labels.shape} dtype={train_labels.dtype}")
    if test_labels is not None:
        print(f"[step] loaded test_labels shape={test_labels.shape} dtype={test_labels.dtype}")

    half = min(args.map_size, args.map_size) / 2.0
    centre = ((args.map_size - 1) / 2.0, (args.map_size - 1) / 2.0)
    mask = sector_mask(
        shape=(args.map_size, args.map_size),
        centre=centre,
        radius_low=args.radius_low_frac * half,
        radius_high=args.radius_high_frac * half,
        angle_range=(args.angle1, args.angle2),
    )

    train_filtered = _extract_first_and_filter(train_arr, args.index, mask)
    test_filtered = _extract_first_and_filter(test_arr, args.index, mask)

    out_dict = {
        "train_phosphenes": train_filtered.astype(np.float32),
        "test_phosphenes": test_filtered.astype(np.float32),
    }

    if train_labels is not None:
        out_dict["train_labels"] = np.atleast_1d(train_labels[args.index]).astype(np.int64)
    if test_labels is not None:
        out_dict["test_labels"] = np.atleast_1d(test_labels[args.index]).astype(np.int64)

    if args.compress:
        np.savez_compressed(args.out_npz, **out_dict)
    else:
        np.savez(args.out_npz, **out_dict)

    print(f"Wrote filtered first sample file: {args.out_npz}")
    print(f"train_phosphenes shape: {train_filtered.shape}, test_phosphenes shape: {test_filtered.shape}")


if __name__ == "__main__":
    main()
