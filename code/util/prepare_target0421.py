from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_TARGETS_DIR = PROJECT_ROOT / "data" / "targets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "targets0421"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "cnnopt" / "baseline" / "analysis"
EMNIST_NPZ = PROJECT_ROOT / "data" / "letters" / "emnist_letters_v3_halfright_128.npz"

MAP_SIZE = 128
N_SELECT = 5


@dataclass(frozen=True)
class RankedSample:
	sample_index: int
	loss: float
	letter_index: int
	letter: str


def _label_to_letter(label: int) -> str:
	if 1 <= label <= 26:
		return chr(ord("A") + label - 1)
	return f"label_{label}"


def _load_ranked_samples(group_name: str, labels: np.ndarray) -> list[RankedSample]:
	group_dir = ANALYSIS_DIR / f"{group_name}_10_by_loss"
	if not group_dir.exists():
		raise FileNotFoundError(f"Missing analysis group directory: {group_dir}")

	params_paths = sorted(group_dir.glob(f"{group_name}_*_sample_*.params.json"))
	if not params_paths:
		raise FileNotFoundError(f"No params json files found in: {group_dir}")

	rows: list[RankedSample] = []
	for path in params_paths:
		with path.open("r", encoding="utf-8") as f:
			data = json.load(f)

		sample_index = int(data["sample_index"])
		loss = float(data["performance"]["loss"])
		if not (0 <= sample_index < len(labels)):
			raise IndexError(f"sample_index out of range in {path}: {sample_index}")

		label_idx = int(labels[sample_index])
		rows.append(
			RankedSample(
				sample_index=sample_index,
				loss=loss,
				letter_index=label_idx,
				letter=_label_to_letter(label_idx),
			)
		)

	return rows


def _pick_unique_letters(rows: list[RankedSample], n_pick: int) -> list[RankedSample]:
	picked: list[RankedSample] = []
	seen_letters: set[str] = set()

	for row in rows:
		if row.letter in seen_letters:
			continue
		seen_letters.add(row.letter)
		picked.append(row)
		if len(picked) == n_pick:
			return picked

	raise RuntimeError(
		f"Could not pick {n_pick} unique letters from candidates. "
		f"Only got {len(picked)} unique letters."
	)


def _resize_map_to_128(map_2d: np.ndarray) -> np.ndarray:
	pil_img = Image.fromarray(map_2d.astype(np.float32), mode="F")
	resized = pil_img.resize((MAP_SIZE, MAP_SIZE), resample=Image.BILINEAR)
	return np.asarray(resized, dtype=np.float32)


def _validate_map(arr: np.ndarray, source_name: str, allow_resize: bool = False) -> np.ndarray:
	map_2d = np.asarray(arr, dtype=np.float32)
	if map_2d.ndim == 3 and map_2d.shape[0] == 1:
		map_2d = map_2d[0]

	if map_2d.shape == (MAP_SIZE, MAP_SIZE):
		return map_2d

	if allow_resize and map_2d.ndim == 2:
		return _resize_map_to_128(map_2d)

	if map_2d.shape != (MAP_SIZE, MAP_SIZE):
		raise ValueError(
			f"{source_name}: expected map shape ({MAP_SIZE}, {MAP_SIZE}), got {map_2d.shape}"
		)
	return map_2d


def _copy_base_targets() -> list[str]:
	base_names = [
		"arc_00.npy",
		"arc_01.npy",
		"arc_02.npy",
		"arc_03.npy",
		"arc_04.npy",
		"single_blob_00.npy",
		"single_blob_01.npy",
		"single_blob_02.npy",
		"single_blob_03.npy",
		"single_blob_04.npy",
		"multi_blob_00.npy",
		"multi_blob_01.npy",
		"multi_blob_02.npy",
		"multi_blob_03.npy",
		"multi_blob_04.npy",
	]

	copied: list[str] = []
	for name in base_names:
		src = SOURCE_TARGETS_DIR / name
		dst = OUTPUT_DIR / name
		if not src.exists():
			raise FileNotFoundError(f"Missing base target file: {src}")

		arr = np.load(src)
		out_map = _validate_map(arr, source_name=str(src), allow_resize=True)
		np.save(dst, out_map.astype(np.float32))
		copied.append(name)

	return copied


def main() -> int:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	copied_base = _copy_base_targets()

	with np.load(EMNIST_NPZ, allow_pickle=False) as data:
		test_labels = data["test_labels"]
		test_maps = data["test_phosphenes"]

		best_candidates = _load_ranked_samples("best", test_labels)
		worst_candidates = _load_ranked_samples("worst", test_labels)

		best_sorted = sorted(best_candidates, key=lambda x: x.loss)
		worst_sorted = sorted(worst_candidates, key=lambda x: x.loss, reverse=True)

		simple_picks = _pick_unique_letters(best_sorted, N_SELECT)
		complex_picks = _pick_unique_letters(worst_sorted, N_SELECT)

		saved_generated: list[str] = []

		for rank, row in enumerate(simple_picks):
			target_map = _validate_map(test_maps[row.sample_index], source_name=f"sample_{row.sample_index:05d}")
			out_name = f"mnist_letter_simple_{rank:02d}_{row.letter}_sample_{row.sample_index:05d}.npy"
			np.save(OUTPUT_DIR / out_name, target_map.astype(np.float32))
			saved_generated.append(out_name)

		for rank, row in enumerate(complex_picks):
			target_map = _validate_map(test_maps[row.sample_index], source_name=f"sample_{row.sample_index:05d}")
			out_name = f"mnist_letter_complex_{rank:02d}_{row.letter}_sample_{row.sample_index:05d}.npy"
			np.save(OUTPUT_DIR / out_name, target_map.astype(np.float32))
			saved_generated.append(out_name)

	summary = {
		"copied_base_files": copied_base,
		"simple_selected": [
			{
				"sample_index": r.sample_index,
				"loss": r.loss,
				"label": r.letter_index,
				"letter": r.letter,
			}
			for r in simple_picks
		],
		"complex_selected": [
			{
				"sample_index": r.sample_index,
				"loss": r.loss,
				"label": r.letter_index,
				"letter": r.letter,
			}
			for r in complex_picks
		],
		"generated_files": saved_generated,
		"total_expected": 25,
		"total_actual": len(list(OUTPUT_DIR.glob("*.npy"))),
	}
	summary_path = OUTPUT_DIR / "selection_summary.json"
	summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	print(f"[done] output dir: {OUTPUT_DIR}")
	print(f"[done] copied base files: {len(copied_base)}")
	print(f"[done] generated letter files: {len(saved_generated)}")
	print(f"[done] total npy files in output: {summary['total_actual']}")
	print(f"[done] summary: {summary_path}")

	if summary["total_actual"] != summary["total_expected"]:
		raise RuntimeError(
			f"Expected {summary['total_expected']} .npy files, got {summary['total_actual']}"
		)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
