from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
	from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
	tqdm = None


INPUT_NPZ = Path(r"D:\yongtae\phosopt\data\letters\emnist_letters_v3_halfright.npz")
OUTPUT_NPZ = Path(r"D:\yongtae\phosopt\data\letters\emnist_letters_v3_halfright_128.npz")
OUTPUT_SIZE = 128


def log(msg: str) -> None:
	now = time.strftime("%H:%M:%S")
	print(f"[{now}] {msg}", flush=True)


def resize_2d_map(map_2d: np.ndarray, output_size: int) -> np.ndarray:
	if map_2d.ndim != 2:
		raise ValueError(f"Expected 2D array, got shape={map_2d.shape}")

	arr = np.asarray(map_2d, dtype=np.float32)
	pil_img = Image.fromarray(arr, mode="F")
	resized = pil_img.resize((output_size, output_size), resample=Image.BILINEAR)
	return np.asarray(resized, dtype=np.float32)


def _resize_job(task: tuple[int, np.ndarray, int]) -> tuple[int, np.ndarray]:
	index, map_2d, output_size = task
	return index, resize_2d_map(map_2d, output_size)


def resize_maps(maps: np.ndarray, key_name: str, output_size: int) -> np.ndarray:
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
	out = np.empty((n, output_size, output_size), dtype=np.float32)

	if n == 0:
		raise ValueError(f"Key '{key_name}' contains zero samples.")

	max_workers = min(os.cpu_count() or 1, 61)
	log(f"  {key_name}: resizing with {max_workers} worker processes")

	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		tasks = ((i, work_maps[i], output_size) for i in range(n))
		if tqdm is not None:
			for index, resized in tqdm(
				executor.map(_resize_job, tasks, chunksize=8),
				total=n,
				desc=f"{key_name}",
				unit="img",
				leave=True,
			):
				out[index] = resized
		else:
			last_print = 0
			for completed_count, (index, resized) in enumerate(executor.map(_resize_job, tasks, chunksize=8), start=1):
				out[index] = resized

				percent = int((completed_count / n) * 100)
				if percent != last_print and (percent % 5 == 0 or completed_count == n):
					log(f"  {key_name}: {completed_count}/{n} ({percent}%)")
					last_print = percent

	if original_ndim == 4 and squeeze_channel:
		out = out[:, None, :, :]

	log(f"Finished key='{key_name}' -> new shape={out.shape}, dtype={out.dtype}")
	return out.astype(np.float32)


def main() -> int:
	log("=== downsize_target.py started ===")
	log(f"Input NPZ : {INPUT_NPZ}")
	log(f"Output NPZ: {OUTPUT_NPZ}")
	log(f"Output size: {OUTPUT_SIZE}x{OUTPUT_SIZE}")

	if not INPUT_NPZ.exists():
		log(f"ERROR: Input NPZ not found: {INPUT_NPZ}")
		return 1

	OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)

	log("Loading NPZ ...")
	with np.load(INPUT_NPZ, allow_pickle=False) as data:
		keys = list(data.files)
		log(f"Found keys: {keys}")

		output_dict: dict[str, Any] = {}
		for key in keys:
			arr = data[key]

			if arr.ndim in (3, 4):
				output_dict[key] = resize_maps(arr, key_name=key, output_size=OUTPUT_SIZE)
			else:
				log(f"Copying key='{key}' without modification (shape={arr.shape}, dtype={arr.dtype})")
				output_dict[key] = arr

	log("Saving downsized NPZ ...")
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
