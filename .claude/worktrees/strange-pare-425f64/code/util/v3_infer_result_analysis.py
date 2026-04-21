# D:\yongtae\phosopt\code\util\v3_infer_result_analysis.py

from __future__ import annotations

import json
import shutil
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
OUTPUT_CORR_TXT = OUTPUT_DIR / "correlation_summary.txt"
OUTPUT_COLLAPSE_TXT = OUTPUT_DIR / "collapse_check.txt"
OUTPUT_CASE_TXT = OUTPUT_DIR / "case_analysis.txt"
OUTPUT_ABLATION_TXT = OUTPUT_DIR / "group_comparison.txt"
OUTPUT_AGGREGATE_TABLE_TXT = OUTPUT_DIR / "aggregate_table_for_paper.txt"

BEST_DIR = OUTPUT_DIR / "best_10_by_loss"
WORST_DIR = OUTPUT_DIR / "worst_10_by_loss"
SCATTER_DIR = OUTPUT_DIR / "scatter_plots"


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


def _format_stat(x: float | None, digits: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def _ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _extract_row_from_sample_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return None

    perf = data.get("performance", {})
    elec = data.get("electrode_stats", {})
    params = data.get("learned_implant_params", {})

    sample_index = data.get("sample_index")
    if sample_index is None:
        try:
            sample_index = int(path.stem.split("_")[1].split(".")[0])
        except Exception:
            sample_index = None

    row: dict[str, Any] = {
        "sample_index": sample_index,
        "source_json": str(path),
    }

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
        row[k] = _safe_float(v)

    return row


def _load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

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

                sample_index = item.get("sample_index")
                if sample_index is None:
                    sample_index = len(rows)

                row: dict[str, Any] = {
                    "sample_index": sample_index,
                    "source_json": str(PER_SAMPLE_JSONL),
                }

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
                    row[k] = _safe_float(v)

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


def _collect_columns(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = sorted({
        k for row in rows for k, v in row.items()
        if isinstance(v, (int, float)) and _safe_float(v) is not None
    })
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
        "cv": float(np.std(arr) / (abs(np.mean(arr)) + 1e-12)),
    }


def _valid_pair(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []

    for row in rows:
        x = _safe_float(row.get(x_key))
        y = _safe_float(row.get(y_key))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)

    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _write_summary_txt(out_path: Path, columns: dict[str, np.ndarray], input_dir: Path) -> None:
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
        "sample_index",
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
        lines.append(f"  cv     : {stats['cv']:.6f}")
        lines.append(f"  min    : {stats['min']:.6f}")
        lines.append(f"  p25    : {stats['p25']:.6f}")
        lines.append(f"  median : {stats['median']:.6f}")
        lines.append(f"  p75    : {stats['p75']:.6f}")
        lines.append(f"  max    : {stats['max']:.6f}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved summary text: {out_path}")


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
        k for k in columns.keys() if k not in preferred_order and k != "sample_index"
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
        ax.axvline(np.mean(arr), linestyle="--", linewidth=1.0)
        ax.axvline(np.median(arr), linestyle=":", linewidth=1.0)
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("Count")

    fig.suptitle("Inference Result Histograms", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[INFO] Saved histogram grid: {save_path}")


def _plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    save_path: Path,
) -> float | None:
    corr = _pearson(x, y)

    plt.figure(figsize=(6.5, 5.5), dpi=150)
    plt.scatter(x, y, s=16, alpha=0.7)

    if x.size >= 2 and np.std(x) > 1e-12:
        coef = np.polyfit(x, y, deg=1)
        xline = np.linspace(np.min(x), np.max(x), 200)
        yline = coef[0] * xline + coef[1]
        plt.plot(xline, yline, linewidth=1.3)

    title = f"{y_label} vs {x_label}"
    if corr is not None:
        title += f" | r={corr:.4f}"
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved scatter: {save_path}")
    return corr


def _run_scatter_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _ensure_clean_dir(SCATTER_DIR)

    pairs = [
        ("num_active_electrodes", "loss"),
        ("num_active_electrodes", "dc_metric"),
        ("num_active_electrodes", "hd_metric"),
    ]

    results: list[dict[str, Any]] = []

    for x_key, y_key in pairs:
        x, y = _valid_pair(rows, x_key, y_key)
        if x.size < 2:
            continue

        save_path = SCATTER_DIR / f"scatter_{y_key}_vs_{x_key}.png"
        corr = _plot_scatter(x, y, x_key, y_key, save_path)
        results.append({
            "x_key": x_key,
            "y_key": y_key,
            "n": int(x.size),
            "pearson_r": corr,
            "save_path": str(save_path),
        })

    return results


def _write_correlation_summary(out_path: Path, scatter_results: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("Scatter / Correlation Summary")
    lines.append("=" * 80)
    lines.append("")

    if not scatter_results:
        lines.append("No valid scatter pairs found.")
    else:
        for item in scatter_results:
            lines.append(f"{item['y_key']} vs {item['x_key']}")
            lines.append(f"  n         : {item['n']}")
            lines.append(f"  pearson_r : {_format_stat(item['pearson_r'], 6)}")
            lines.append(f"  file      : {item['save_path']}")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved correlation summary: {out_path}")


def _get_sample_file_paths(input_dir: Path, sample_index: int) -> dict[str, Path]:
    stem = f"sample_{sample_index:05d}"
    return {
        "params_json": input_dir / f"{stem}.params.json",
        "electrodes_npy": input_dir / f"{stem}.electrodes.npy",
        "recon_npy": input_dir / f"{stem}.recon.npy",
        "target_npy": input_dir / f"{stem}.target.npy",
        "recon_png": input_dir / f"{stem}.recon.png",
        "target_png": input_dir / f"{stem}.target.png",
    }


def _copy_case_files(case_rows: list[dict[str, Any]], dst_dir: Path, prefix: str) -> None:
    _ensure_clean_dir(dst_dir)

    for rank, row in enumerate(case_rows, start=1):
        sample_index = row.get("sample_index")
        if sample_index is None:
            continue

        srcs = _get_sample_file_paths(INPUT_DIR, int(sample_index))
        meta_path = dst_dir / f"{prefix}_{rank:02d}_sample_{int(sample_index):05d}_metrics.txt"

        lines = [
            f"rank={rank}",
            f"sample_index={sample_index}",
            f"dc_metric={_format_stat(_safe_float(row.get('dc_metric')), 6)}",
            f"hd_metric={_format_stat(_safe_float(row.get('hd_metric')), 6)}",
            f"y_metric={_format_stat(_safe_float(row.get('y_metric')), 6)}",
            f"score={_format_stat(_safe_float(row.get('score')), 6)}",
            f"loss={_format_stat(_safe_float(row.get('loss')), 6)}",
            f"num_active_electrodes={_format_stat(_safe_float(row.get('num_active_electrodes')), 6)}",
            f"active_ratio={_format_stat(_safe_float(row.get('active_ratio')), 6)}",
            f"mean_activation={_format_stat(_safe_float(row.get('mean_activation')), 6)}",
            f"alpha={_format_stat(_safe_float(row.get('alpha')), 6)}",
            f"beta={_format_stat(_safe_float(row.get('beta')), 6)}",
            f"offset_from_base={_format_stat(_safe_float(row.get('offset_from_base')), 6)}",
            f"shank_length={_format_stat(_safe_float(row.get('shank_length')), 6)}",
        ]
        meta_path.write_text("\n".join(lines), encoding="utf-8")

        if srcs["target_png"].exists():
            _copy_if_exists(
                srcs["target_png"],
                dst_dir / f"{prefix}_{rank:02d}_sample_{int(sample_index):05d}_target.png",
            )
        if srcs["recon_png"].exists():
            _copy_if_exists(
                srcs["recon_png"],
                dst_dir / f"{prefix}_{rank:02d}_sample_{int(sample_index):05d}_recon.png",
            )
        if srcs["params_json"].exists():
            _copy_if_exists(
                srcs["params_json"],
                dst_dir / f"{prefix}_{rank:02d}_sample_{int(sample_index):05d}.params.json",
            )


def _summarize_group(rows: list[dict[str, Any]], group_name: str) -> list[str]:
    keys = [
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

    lines = [f"[{group_name}]"]
    lines.append(f"n = {len(rows)}")

    for key in keys:
        arr = np.asarray(
            [v for v in (_safe_float(r.get(key)) for r in rows) if v is not None],
            dtype=np.float64,
        )
        if arr.size == 0:
            continue
        lines.append(
            f"  {key}: mean={np.mean(arr):.6f}, std={np.std(arr):.6f}, "
            f"median={np.median(arr):.6f}, min={np.min(arr):.6f}, max={np.max(arr):.6f}"
        )
    lines.append("")
    return lines


def _run_case_analysis(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_rows = [r for r in rows if _safe_float(r.get("loss")) is not None]
    if not valid_rows:
        return [], []

    sorted_by_loss = sorted(valid_rows, key=lambda r: float(r["loss"]))
    best_10 = sorted_by_loss[:10]
    worst_10 = sorted_by_loss[-10:]

    _copy_case_files(best_10, BEST_DIR, "best")
    _copy_case_files(worst_10, WORST_DIR, "worst")

    lines: list[str] = []
    lines.append("Case Analysis")
    lines.append("=" * 80)
    lines.append("")
    lines.extend(_summarize_group(best_10, "BEST 10 BY LOSS"))
    lines.extend(_summarize_group(worst_10, "WORST 10 BY LOSS"))

    lines.append("[BEST 10 SAMPLE LIST]")
    for r in best_10:
        lines.append(
            f"sample_{int(r['sample_index']):05d}: "
            f"loss={_format_stat(_safe_float(r.get('loss')))}, "
            f"dc={_format_stat(_safe_float(r.get('dc_metric')))}, "
            f"hd={_format_stat(_safe_float(r.get('hd_metric')))}, "
            f"active={_format_stat(_safe_float(r.get('num_active_electrodes')))}"
        )
    lines.append("")

    lines.append("[WORST 10 SAMPLE LIST]")
    for r in worst_10:
        lines.append(
            f"sample_{int(r['sample_index']):05d}: "
            f"loss={_format_stat(_safe_float(r.get('loss')))}, "
            f"dc={_format_stat(_safe_float(r.get('dc_metric')))}, "
            f"hd={_format_stat(_safe_float(r.get('hd_metric')))}, "
            f"active={_format_stat(_safe_float(r.get('num_active_electrodes')))}"
        )
    lines.append("")

    OUTPUT_CASE_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved case analysis: {OUTPUT_CASE_TXT}")

    return best_10, worst_10


def _run_group_comparison(rows: list[dict[str, Any]]) -> None:
    valid_active = [r for r in rows if _safe_float(r.get("num_active_electrodes")) is not None]
    valid_dc = [r for r in rows if _safe_float(r.get("dc_metric")) is not None]

    lines: list[str] = []
    lines.append("Group Comparison / Ablation-like Interpretation")
    lines.append("=" * 80)
    lines.append("")

    if valid_active:
        sorted_active = sorted(valid_active, key=lambda r: float(r["num_active_electrodes"]))
        n = len(sorted_active)
        k = max(1, n // 4)
        low_active = sorted_active[:k]
        high_active = sorted_active[-k:]
        lines.extend(_summarize_group(low_active, f"LOW ACTIVE ELECTRODES (bottom 25%, n={len(low_active)})"))
        lines.extend(_summarize_group(high_active, f"HIGH ACTIVE ELECTRODES (top 25%, n={len(high_active)})"))

    if valid_dc:
        sorted_dc = sorted(valid_dc, key=lambda r: float(r["dc_metric"]), reverse=True)
        n = len(sorted_dc)
        k = max(1, n // 4)
        high_dc = sorted_dc[:k]
        low_dc = sorted_dc[-k:]
        lines.extend(_summarize_group(high_dc, f"HIGH DC (top 25%, n={len(high_dc)})"))
        lines.extend(_summarize_group(low_dc, f"LOW DC (bottom 25%, n={len(low_dc)})"))

    OUTPUT_ABLATION_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved group comparison: {OUTPUT_ABLATION_TXT}")


def _run_collapse_check(columns: dict[str, np.ndarray]) -> None:
    lines: list[str] = []
    lines.append("Collapse / Low-Variation Check")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Heuristic:")
    lines.append("  - std very small or cv very small means outputs may be nearly constant.")
    lines.append("  - especially important for num_active_electrodes and 4 implant params.")
    lines.append("")

    check_keys = [
        "num_active_electrodes",
        "active_ratio",
        "mean_activation",
        "alpha",
        "beta",
        "offset_from_base",
        "shank_length",
    ]

    suspicious: list[str] = []

    for key in check_keys:
        arr = columns.get(key)
        if arr is None or arr.size == 0:
            continue

        stats = _describe(arr)
        unique_rounded = len(np.unique(np.round(arr, 6)))
        dynamic_range = float(np.max(arr) - np.min(arr))

        lines.append(f"[{key}]")
        lines.append(f"  mean         : {stats['mean']:.6f}")
        lines.append(f"  std          : {stats['std']:.6f}")
        lines.append(f"  cv           : {stats['cv']:.6f}")
        lines.append(f"  min          : {stats['min']:.6f}")
        lines.append(f"  max          : {stats['max']:.6f}")
        lines.append(f"  range        : {dynamic_range:.6f}")
        lines.append(f"  unique(~1e-6): {unique_rounded}")

        is_suspicious = False
        reason_parts: list[str] = []

        if stats["std"] < 1e-6:
            is_suspicious = True
            reason_parts.append("std < 1e-6")
        if stats["cv"] < 1e-3:
            is_suspicious = True
            reason_parts.append("cv < 1e-3")
        if unique_rounded <= 3:
            is_suspicious = True
            reason_parts.append("unique values <= 3")

        if is_suspicious:
            reason = ", ".join(reason_parts)
            lines.append(f"  verdict      : SUSPICIOUS ({reason})")
            suspicious.append(f"{key}: {reason}")
        else:
            lines.append("  verdict      : variation exists")

        lines.append("")

    lines.append("[Overall interpretation]")
    if suspicious:
        lines.append("Potential collapse-like signs detected in:")
        for s in suspicious:
            lines.append(f"  - {s}")
    else:
        lines.append("No strong collapse-like sign detected by the simple variance heuristics.")

    OUTPUT_COLLAPSE_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved collapse check: {OUTPUT_COLLAPSE_TXT}")


def _write_aggregate_table(columns: dict[str, np.ndarray], out_path: Path) -> None:
    metrics = [
        "dc_metric",
        "hd_metric",
        "y_metric",
        "num_active_electrodes",
        "active_ratio",
        "mean_activation",
        "score",
        "loss",
    ]

    lines: list[str] = []
    lines.append("Aggregate Table For Paper")
    lines.append("=" * 80)
    lines.append("")
    lines.append("metric\tmean\tstd")

    for m in metrics:
        arr = columns.get(m)
        if arr is None or arr.size == 0:
            continue
        lines.append(f"{m}\t{np.mean(arr):.6f}\t{np.std(arr):.6f}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Saved aggregate table: {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("[INFO] Starting v3 inference result analysis")
    print(f"[INFO] Input directory : {INPUT_DIR}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_clean_dir(BEST_DIR)
    _ensure_clean_dir(WORST_DIR)
    _ensure_clean_dir(SCATTER_DIR)

    rows = _load_rows(INPUT_DIR)
    print(f"[INFO] Loaded {len(rows)} per-sample rows")

    columns = _collect_columns(rows)
    if not columns:
        raise RuntimeError("No numeric columns were collected from the inference outputs.")

    # 1. Overall summary txt
    _write_summary_txt(
        out_path=OUTPUT_SUMMARY_TXT,
        columns=columns,
        input_dir=INPUT_DIR,
    )

    # 2. Histogram grid png only
    _plot_grid_hists(columns, OUTPUT_GRID_PNG)

    # 3. Scatter analysis: only 3 plots
    scatter_results = _run_scatter_analysis(rows)
    _write_correlation_summary(OUTPUT_CORR_TXT, scatter_results)

    # 4. Case analysis: by loss
    _run_case_analysis(rows)

    # 5. Collapse check txt
    _run_collapse_check(columns)

    # 6. Aggregate table txt
    _write_aggregate_table(columns, OUTPUT_AGGREGATE_TABLE_TXT)

    # 7. Group comparison txt
    _run_group_comparison(rows)

    print("[INFO] Analysis complete.")


if __name__ == "__main__":
    main()