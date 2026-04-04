# D:\yongtae\phosopt\code\util\make_mnist_small.py

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


INPUT_NPZ = Path(r"D:\yongtae\phosopt\data\letters\emnist_letters_phosphenes.npz")
OUTPUT_NPZ = Path(r"D:\yongtae\phosopt\data\letters\emnist_letters_v3_halfright.npz")
RESULTS_DIR = Path(r"D:\yongtae\phosopt\results")
PREVIEW_PNG = RESULTS_DIR / "emnist_letters_v3_halfright_first.png"


def log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def apply_right_semicircle_mask(image_2d: np.ndarray) -> np.ndarray:
    """
    전체 이미지 기준 내접원을 만들고,
    그 중 오른쪽 반원 내부만 남기고 나머지는 0으로 지웁니다.
    """
    if image_2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={image_2d.shape}")

    arr = np.asarray(image_2d, dtype=np.float32)
    h, w = arr.shape

    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = min(h, w) / 2.0

    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2

    inside_circle = dist2 <= radius ** 2
    right_half = xx >= cx
    mask = inside_circle & right_half

    out = np.zeros_like(arr, dtype=np.float32)
    out[mask] = arr[mask]
    return out


def process_single_map(map_2d: np.ndarray) -> np.ndarray:
    """
    처리 순서:
      1. 절반 크기로 축소
      2. 오른쪽 정렬
      3. 전체 이미지 기준 오른쪽 반원 마스크 적용
    """
    if map_2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={map_2d.shape}")

    arr = np.asarray(map_2d, dtype=np.float32)
    h, w = arr.shape

    new_h = max(1, h // 2)
    new_w = max(1, w // 2)

    pil_img = Image.fromarray(arr, mode="F")
    resized = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)
    small = np.array(resized, dtype=np.float32)

    canvas = np.zeros((h, w), dtype=np.float32)

    # 오른쪽 정렬 + 세로 중앙 정렬
    y0 = (h - new_h) // 2
    x0 = w - new_w
    canvas[y0:y0 + new_h, x0:x0 + new_w] = small

    masked = apply_right_semicircle_mask(canvas)
    return masked.astype(np.float32)


def save_preview_png(first_image: np.ndarray, png_path: Path) -> None:
    """
    첫 이미지를 PNG로 저장합니다.
    보기 좋게 [0,255] min-max 정규화해서 저장합니다.
    """
    png_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(first_image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Preview image must be 2D or [1,H,W], got {arr.shape}")

    min_val = float(arr.min())
    max_val = float(arr.max())

    if max_val > min_val:
        norm = (arr - min_val) / (max_val - min_val)
    else:
        norm = np.zeros_like(arr, dtype=np.float32)

    img_u8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_u8, mode="L").save(png_path)
    log(f"Saved preview PNG: {png_path}")


def transform_maps_with_first_preview(
    maps: np.ndarray,
    key_name: str,
    preview_png_path: Path | None = None,
    save_preview: bool = False,
) -> np.ndarray:
    """
    [N,H,W] 또는 [N,1,H,W]를 처리합니다.

    순서:
      1. 첫 이미지 먼저 처리
      2. 첫 이미지 PNG 저장
      3. 나머지 이미지 처리
    """
    log(f"Processing key='{key_name}' ... original shape={maps.shape}, dtype={maps.dtype}")

    original_ndim = maps.ndim

    if maps.ndim == 4 and maps.shape[1] == 1:
        squeeze_channel = True
        work_maps = maps[:, 0, :, :]
    elif maps.ndim == 3:
        squeeze_channel = False
        work_maps = maps
    else:
        raise ValueError(
            f"Key '{key_name}' must have shape [N,H,W] or [N,1,H,W], got {maps.shape}"
        )

    n = work_maps.shape[0]
    out = np.empty_like(work_maps, dtype=np.float32)

    if n == 0:
        raise ValueError(f"Key '{key_name}' contains zero samples.")

    log(f"  {key_name}: processing first image for preview ...")
    out[0] = process_single_map(work_maps[0])

    if save_preview:
        if preview_png_path is None:
            raise ValueError("preview_png_path must be provided when save_preview=True")
        save_preview_png(out[0], preview_png_path)

    log(f"  {key_name}: first image done (1/{n})")

    last_print = 0
    for i in range(1, n):
        out[i] = process_single_map(work_maps[i])

        percent = int(((i + 1) / n) * 100)
        if percent != last_print and (percent % 5 == 0 or i == n - 1):
            log(f"  {key_name}: {i + 1}/{n} ({percent}%)")
            last_print = percent

    if original_ndim == 4 and squeeze_channel:
        out = out[:, None, :, :]

    log(f"Finished key='{key_name}' -> new shape={out.shape}, dtype={out.dtype}")
    return out.astype(np.float32)


def main() -> int:
    log("=== make_mnist_small.py started ===")
    log(f"Input NPZ : {INPUT_NPZ}")
    log(f"Output NPZ: {OUTPUT_NPZ}")
    log(f"Preview   : {PREVIEW_PNG}")

    if not INPUT_NPZ.exists():
        log(f"ERROR: Input NPZ not found: {INPUT_NPZ}")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)

    log("Loading NPZ ...")
    with np.load(INPUT_NPZ, allow_pickle=False) as data:
        keys = list(data.files)
        log(f"Found keys: {keys}")

        output_dict: dict[str, Any] = {}
        preview_already_saved = False

        for key in keys:
            arr = data[key]

            if arr.ndim in (3, 4):
                transformed = transform_maps_with_first_preview(
                    maps=arr,
                    key_name=key,
                    preview_png_path=PREVIEW_PNG,
                    save_preview=not preview_already_saved,
                )
                output_dict[key] = transformed

                if not preview_already_saved:
                    preview_already_saved = True
            else:
                log(f"Copying key='{key}' without modification (shape={arr.shape}, dtype={arr.dtype})")
                output_dict[key] = arr

    log("Saving new NPZ ...")
    np.savez_compressed(OUTPUT_NPZ, **output_dict)
    log(f"Saved output NPZ: {OUTPUT_NPZ}")

    log("Verifying saved NPZ ...")
    with np.load(OUTPUT_NPZ, allow_pickle=False) as check:
        for key in check.files:
            log(f"  verified key='{key}' shape={check[key].shape}, dtype={check[key].dtype}")

    log("=== done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())