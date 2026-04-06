# D:\yongtae\phosopt\code\util\v3_infer_result_analysis.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = Path(r"D:\yongtae\phosopt\data\output\infer_v3_halfright")
OUTPUT_DIR = Path(r"D:\yongtae\phosopt\data\output\infer_v3_analysis")

PER_SAMPLE_JSONL = INPUT_DIR / "per_sample_metrics.jsonl"
SUMMARY_JSON = INPUT_DIR / "summary.json"

OUTPUT_SUMMARY_TXT = OUTPUT_DIR / "overall_summary.txt"
OUTPUT_GRID_PNG = OUTPUT_DIR / "hist_all_metrics_grid.png"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _extract_row_from_sample_json(path: Path) -> dict[str, float] | None:
    """
    Fallback parser for sample_*.params.json files.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return None

    perf = data.get("performance", {})
    elec = data.get("electrode_stats", {})
    params = data.get("learned_implant_params", {})

    row: dict[str, float] = {}

    mapping = {
        "dc_metric": perf.get("dc_metric"),
        "y_metric": perf.get("y_metric"),
        "hd_metric": perf.get("hd_metric"),
        "score": perf.get("score"),
        "loss": perf.get("loss"),
        "num_active_electrodes": elec.get("num_active_electrodes"),
        "active_ratio": elec.get("active_ratio"),
        "mean_activation": elec.get("mean_activation"),
        "alpha": params.get("alpha"),
        "beta": params.get("beta"),
        "offset_from_base": params.get("offset_from_base"),
        "shank_length": params.get("shank_length"),
    }

    for k, v in mapping.items():
        fv = _safe_float(v)
        if fv is not None:
            row[k] = fv

    return row if row else None


def _load_rows(input_dir: Path) -> list[dict[str, float]]:
    """
    Preferred source:
      - per_sample_metrics.jsonl

    Fallback:
      - sample_*.params.json
    """
    rows: list[dict[str, float]] = []

    if PER_SAMPLE_JSONL.exists():
        print(f"[INFO] Loading per-sample metrics from JSONL: {PER_SAMPLE_JSONL}")
        with PER_SAMPLE_JSONL.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception as e:
                    print(f"[WARN] Failed to parse JSONL line {line_idx}: {e}")
                    continue

                perf = item.get("performance", {})
                elec = item.get("electrode_stats", {})
                params = item.get("learned_implant_params", {})

                row: dict[str, float] = {}

                mapping = {
                    "dc_metric": perf.get("dc_metric"),
                    "y_metric": perf.get("y_metric"),
                    "hd_metric": perf.get("hd_metric"),
                    "score": perf.get("score"),
                    "loss": perf.get("loss"),
                    "num_active_electrodes": elec.get("num_active_electrodes"),
                    "active_ratio": elec.get("active_ratio"),
                    "mean_activation": elec.get("mean_activation"),
                    "alpha": params.get("alpha"),
                    "beta": params.get("beta"),
                    "offset_from_base": params.get("offset_from_base"),
                    "shank_length": params.get("shank_length"),
                }

                for k, v in mapping.items():
                    fv = _safe_float(v)
                    if fv is not None:
                        row[k] = fv

                if row:
                    rows.append(row)

        if rows:
            return rows
        print("[WARN] JSONL existed but no valid rows were loaded. Falling back to sample JSON files.")

    sample_jsons = sorted(input_dir.glob("sample_*.params.json"))
    if not sample_jsons:
        raise FileNotFoundError(
            f"No per-sample result files found in {input_dir}\n"
            f"Expected either:\n"
            f"  - {PER_SAMPLE_JSONL.name}\n"
            f"  - sample_*.params.json"
        )

    print(f"[INFO] Loading fallback sample JSON files: {len(sample_jsons)} files")
    for path in sample_jsons:
        row = _extract_row_from_sample_json(path)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("No valid per-sample rows could be loaded.")

    return rows


def _collect_columns(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    keys = sorted({k for row in rows for k in row.keys()})
    columns: dict[str, np.ndarray] = {}

    for key in keys:
        vals = [_safe_float(row.get(key)) for row in rows]
        arr = np.asarray([v for v in vals if v is not None], dtype=np.float64)
        if arr.size > 0:
            columns[key] = arr

    return columns


def _describe(arr: np.ndarray) -> dict[str, float]:
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def _write_summary_txt(
    out_path: Path,
    columns: dict[str, np.ndarray],
    input_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("PhosOpt Inference Result Analysis")
    lines.append("=" * 80)
    lines.append(f"Input directory: {input_dir}")
    lines.append(f"Number of metrics: {len(columns)}")
    lines.append("")

    if SUMMARY_JSON.exists():
        lines.append(f"Found aggregate summary.json: {SUMMARY_JSON}")
        try:
            summary_data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
            lines.append("summary.json basic info:")
            lines.append(f"  num_samples: {summary_data.get('num_samples')}")
            lines.append(f"  emnist_split: {summary_data.get('emnist_split')}")
            lines.append(f"  hemisphere: {summary_data.get('hemisphere')}")
            lines.append(f"  map_size: {summary_data.get('map_size')}")
            lines.append("")
        except Exception as e:
            lines.append(f"[WARN] Failed to parse summary.json: {e}")
            lines.append("")

    preferred_order = [
        "dc_metric",
        "hd_metric",
        "y_metric",
        "score",
        "loss",
        "num_active_electrodes",
        "active_ratio",
        "mean_activation",
        "alpha",
        "beta",
        "offset_from_base",
        "shank_length",
    ]

    ordered_keys = [k for k in preferred_order if k in columns] + [
        k for k in columns.keys() if k not in preferred_order
    ]

    for key in ordered_keys:
        arr = columns[key]
        stats = _describe(arr)

        lines.append(f"[{key}]")
        lines.append(f"  count  : {int(stats['count'])}")
        lines.append(f"  mean   : {stats['mean']:.6f}")
        lines.append(f"  std    : {stats['std']:.6f}")
        lines.append(f"  min    : {stats['min']:.6f}")
        lines.append(f"  p25    : {stats['p25']:.6f}")
        lines.append(f"  median : {stats['median']:.6f}")
        lines.append(f"  p75    : {stats['p75']:.6f}")
        lines.append(f"  max    : {stats['max']:.6f}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved summary text: {out_path}")


def _plot_single_hist(metric_name: str, arr: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(7, 5), dpi=150)
    plt.hist(arr, bins=30)
    plt.title(f"Histogram: {metric_name}")
    plt.xlabel(metric_name)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved histogram: {save_path}")


def _plot_grid_hists(columns: dict[str, np.ndarray], save_path: Path) -> None:
    preferred_order = [
        "dc_metric",
        "hd_metric",
        "y_metric",
        "score",
        "loss",
        "num_active_electrodes",
        "active_ratio",
        "mean_activation",
        "alpha",
        "beta",
        "offset_from_base",
        "shank_length",
    ]

    metric_names = [k for k in preferred_order if k in columns] + [
        k for k in columns.keys() if k not in preferred_order
    ]

    n = len(metric_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.8 * nrows), dpi=150)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes.flat:
        ax.axis("off")

    for idx, name in enumerate(metric_names):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        ax.axis("on")

        arr = columns[name]
        ax.hist(arr, bins=30)
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("Count")

    fig.suptitle("Inference Result Histograms", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[INFO] Saved histogram grid: {save_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("[INFO] Starting v3 inference result analysis")
    print(f"[INFO] Input directory : {INPUT_DIR}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(INPUT_DIR)
    print(f"[INFO] Loaded {len(rows)} per-sample rows")

    columns = _collect_columns(rows)
    if not columns:
        raise RuntimeError("No numeric columns were collected from the inference outputs.")

    _write_summary_txt(
        out_path=OUTPUT_SUMMARY_TXT,
        columns=columns,
        input_dir=INPUT_DIR,
    )

    for metric_name, arr in columns.items():
        save_path = OUTPUT_DIR / f"hist_{metric_name}.png"
        _plot_single_hist(metric_name, arr, save_path)

    _plot_grid_hists(columns, OUTPUT_GRID_PNG)

    print("[INFO] Analysis complete.")


if __name__ == "__main__":
    main()