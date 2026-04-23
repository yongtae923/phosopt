from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_DIR = PROJECT_ROOT / "data" / "arena0422" / "comparison"
COMPARISON_DIR = Path(os.getenv("PHOSOPT_ARENA_COMPARISON_DIR", str(DEFAULT_COMPARISON_DIR)))

INPUT_CSV = COMPARISON_DIR / "per_target_metrics.csv"
OUTPUT_LETTERS_PNG = COMPARISON_DIR / "plot_strategy_mean_loss_letters.png"
OUTPUT_NON_LETTERS_PNG = COMPARISON_DIR / "plot_strategy_mean_loss_non_letters.png"


def _is_letter_target(target_name: str) -> bool:
	return "letter" in target_name.lower()


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
	if not csv_path.exists():
		raise FileNotFoundError(f"Comparison CSV not found: {csv_path}")

	rows: list[dict[str, str]] = []
	with csv_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if row:
				rows.append(row)
	if not rows:
		raise RuntimeError(f"No rows loaded from CSV: {csv_path}")
	return rows


def _group_mean_loss(rows: list[dict[str, str]], want_letters: bool) -> dict[str, float]:
	grouped: dict[str, list[float]] = defaultdict(list)
	for row in rows:
		target_name = str(row.get("target_name", ""))
		is_letter = _is_letter_target(target_name)
		if is_letter != want_letters:
			continue

		strategy = str(row.get("strategy", "")).strip()
		if not strategy:
			continue

		try:
			loss = float(row.get("loss", ""))
		except Exception:
			continue
		grouped[strategy].append(loss)

	return {k: float(np.mean(v)) for k, v in grouped.items() if v}


def _save_mean_loss_plot(mean_loss: dict[str, float], save_path: Path, title: str) -> None:
	if not mean_loss:
		print(f"[WARN] Skip plot (no data): {save_path}")
		return

	strategies = sorted(mean_loss.keys())
	values = [mean_loss[s] for s in strategies]
	x = np.arange(len(strategies))

	fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
	bars = ax.bar(x, values)
	ax.set_xticks(x)
	ax.set_xticklabels(strategies, rotation=25, ha="right")
	ax.set_ylabel("Mean Loss")
	ax.set_title(title)
	ax.grid(axis="y", alpha=0.25)

	for b, v in zip(bars, values):
		ax.text(
			b.get_x() + b.get_width() / 2.0,
			b.get_height(),
			f"{v:.3f}",
			ha="center",
			va="bottom",
			fontsize=8,
		)

	fig.tight_layout()
	fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
	plt.close(fig)
	print(f"[INFO] Saved plot: {save_path}")


def main() -> None:
	print(f"[INFO] Comparison directory: {COMPARISON_DIR}")
	rows = _load_rows(INPUT_CSV)

	letters_mean = _group_mean_loss(rows, want_letters=True)
	non_letters_mean = _group_mean_loss(rows, want_letters=False)

	_save_mean_loss_plot(
		letters_mean,
		OUTPUT_LETTERS_PNG,
		title="Mean Loss by Strategy (Letter Targets)",
	)
	_save_mean_loss_plot(
		non_letters_mean,
		OUTPUT_NON_LETTERS_PNG,
		title="Mean Loss by Strategy (Non-letter Targets)",
	)

	print("[INFO] Done.")


if __name__ == "__main__":
	main()
