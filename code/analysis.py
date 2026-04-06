# D:\yongtae\phosopt\code\analysis.py

"""
Run v3 inference result analysis for four ablation variants.

This script reuses code/util/v3_infer_result_analysis.py by redirecting its
input/output globals per variant, then runs it sequentially.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "output"

VARIANTS = [
    "v3_1_baseline",
    "v3_2_no_sparsity",
    "v3_3_no_invalid_region",
    "v3_4_no_aux",
]

SCRIPT_PATH = PROJECT_ROOT / "code" / "util" / "v3_infer_result_analysis.py"
AGGREGATE_JSON = OUTPUT_ROOT / "analysis_variants_summary.json"
AGGREGATE_TXT = OUTPUT_ROOT / "analysis_variants_summary.txt"
AGGREGATE_CSV = OUTPUT_ROOT / "analysis_variants_summary.csv"


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("v3_infer_result_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_module(mod: ModuleType, input_dir: Path, output_dir: Path) -> None:
    mod.INPUT_DIR = input_dir
    mod.OUTPUT_DIR = output_dir

    mod.PER_SAMPLE_JSONL = input_dir / "per_sample_metrics.jsonl"
    mod.SUMMARY_JSON = input_dir / "summary.json"

    mod.OUTPUT_SUMMARY_TXT = output_dir / "overall_summary.txt"
    mod.OUTPUT_GRID_PNG = output_dir / "hist_all_metrics_grid.png"
    mod.OUTPUT_CORR_TXT = output_dir / "correlation_summary.txt"
    mod.OUTPUT_COLLAPSE_TXT = output_dir / "collapse_check.txt"
    mod.OUTPUT_CASE_TXT = output_dir / "case_analysis.txt"
    mod.OUTPUT_ABLATION_TXT = output_dir / "group_comparison.txt"
    mod.OUTPUT_AGGREGATE_TABLE_TXT = output_dir / "aggregate_table_for_paper.txt"

    mod.BEST_DIR = output_dir / "best_10_by_loss"
    mod.WORST_DIR = output_dir / "worst_10_by_loss"
    mod.SCATTER_DIR = output_dir / "scatter_plots"


def _safe_get_summary_means(summary_path: Path) -> dict[str, float | None]:
    if not summary_path.exists():
        return {
            "dc_mean": None,
            "y_mean": None,
            "hd_mean": None,
            "score_mean": None,
            "loss_mean": None,
            "active_mean": None,
        }

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    perf = data.get("aggregate_performance", {})
    elec = data.get("aggregate_electrode_stats", {})

    return {
        "dc_mean": perf.get("dc_metric", {}).get("mean"),
        "y_mean": perf.get("y_metric", {}).get("mean"),
        "hd_mean": perf.get("hd_metric", {}).get("mean"),
        "score_mean": perf.get("score", {}).get("mean"),
        "loss_mean": perf.get("loss", {}).get("mean"),
        "active_mean": elec.get("num_active_electrodes", {}).get("mean"),
    }


def _write_summary_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "variant",
        "status",
        "infer_dir",
        "analysis_dir",
        "dc_mean",
        "y_mean",
        "hd_mean",
        "score_mean",
        "loss_mean",
        "active_mean",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    print("[analysis] Starting multi-variant analysis")
    print(f"[analysis] Output root: {OUTPUT_ROOT}")

    v3 = _load_module(SCRIPT_PATH)

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        infer_dir = OUTPUT_ROOT / f"infer_{variant}"
        analysis_dir = OUTPUT_ROOT / f"analysis_{variant}"

        print("\n" + "=" * 80)
        print(f"[analysis] Variant: {variant}")
        print(f"[analysis] Infer input : {infer_dir}")
        print(f"[analysis] Analysis out: {analysis_dir}")
        print("=" * 80)

        if not infer_dir.exists():
            print(f"[analysis][WARN] Missing infer directory. Skipping: {infer_dir}")
            rows.append(
                {
                    "variant": variant,
                    "status": "skipped_missing_infer_dir",
                    "infer_dir": str(infer_dir),
                    "analysis_dir": str(analysis_dir),
                    **_safe_get_summary_means(infer_dir / "summary.json"),
                }
            )
            continue

        try:
            _configure_module(v3, input_dir=infer_dir, output_dir=analysis_dir)
            v3.main()
            status = "ok"
        except Exception as e:
            print(f"[analysis][ERROR] Failed variant {variant}: {e}")
            status = f"failed: {type(e).__name__}: {e}"

        rows.append(
            {
                "variant": variant,
                "status": status,
                "infer_dir": str(infer_dir),
                "analysis_dir": str(analysis_dir),
                **_safe_get_summary_means(infer_dir / "summary.json"),
            }
        )

    AGGREGATE_JSON.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")

    lines = []
    lines.append("PhosOpt Variant Analysis Summary")
    lines.append("=" * 80)
    lines.append("")
    for row in rows:
        lines.append(f"variant: {row['variant']}")
        lines.append(f"  status    : {row['status']}")
        lines.append(f"  dc_mean   : {row['dc_mean']}")
        lines.append(f"  y_mean    : {row['y_mean']}")
        lines.append(f"  hd_mean   : {row['hd_mean']}")
        lines.append(f"  score_mean: {row['score_mean']}")
        lines.append(f"  loss_mean : {row['loss_mean']}")
        lines.append(f"  active_mean: {row['active_mean']}")
        lines.append("")

    AGGREGATE_TXT.write_text("\n".join(lines), encoding="utf-8")
    _write_summary_csv(rows, AGGREGATE_CSV)

    print("\n[analysis] Completed multi-variant analysis")
    print(f"[analysis] Saved: {AGGREGATE_JSON}")
    print(f"[analysis] Saved: {AGGREGATE_TXT}")
    print(f"[analysis] Saved: {AGGREGATE_CSV}")


if __name__ == "__main__":
    main()
