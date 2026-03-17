"""
Result analysis script for PhosOpt experiments.

Reads JSON-Lines log files produced by the benchmark runners and generates:
  - Summary tables (mean +/- std per method, grouped by benchmark)
  - Score comparison bar charts
  - Efficiency plots (score vs simulator calls, score vs wall time)
  - Statistical tests (Wilcoxon signed-rank between PhosOpt and each baseline)

Usage:
    python code/analyze.py --results-dir results/
    python code/analyze.py --results-dir results/per_target --benchmark per_target
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_logs(results_dir: Path) -> pd.DataFrame:
    """Load all .jsonl files under *results_dir* into a single DataFrame."""
    rows: list[dict] = []
    for jl in results_dir.rglob("*.jsonl"):
        with open(jl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        print(f"No results found in {results_dir}")
        sys.exit(1)
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame, benchmark: str | None = None) -> pd.DataFrame:
    """Mean +/- std of key metrics grouped by method."""
    if benchmark:
        df = df[df["benchmark_type"] == benchmark]
    metrics = ["loss", "score", "dc", "y", "hd", "active_electrode_count", "simulator_calls", "wall_clock_time"]
    available = [m for m in metrics if m in df.columns]

    grouped = df.groupby("method")[available].agg(["mean", "std"])
    # Flatten multi-level columns
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    return grouped.round(4)


def plot_scores(df: pd.DataFrame, output_dir: Path, benchmark: str | None = None) -> None:
    """Bar chart of mean score by method."""
    if benchmark:
        df = df[df["benchmark_type"] == benchmark]
    means = df.groupby("method")["score"].mean()
    stds = df.groupby("method")["score"].std().fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    means.plot.bar(ax=ax, yerr=stds, capsize=4)
    ax.set_ylabel("Composite Score (S)")
    ax.set_title(f"Score Comparison{f' ({benchmark})' if benchmark else ''}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / f"score_comparison{'_' + benchmark if benchmark else ''}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: score_comparison{'_' + benchmark if benchmark else ''}.png")


def plot_efficiency(df: pd.DataFrame, output_dir: Path, benchmark: str | None = None) -> None:
    """Score vs simulator calls scatter."""
    if benchmark:
        df = df[df["benchmark_type"] == benchmark]
    if "simulator_calls" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for method, group in df.groupby("method"):
        ax.scatter(group["simulator_calls"], group["score"], label=method, alpha=0.7, s=30)
    ax.set_xlabel("Simulator Calls")
    ax.set_ylabel("Score (S)")
    ax.set_title(f"Efficiency{f' ({benchmark})' if benchmark else ''}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"efficiency{'_' + benchmark if benchmark else ''}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: efficiency{'_' + benchmark if benchmark else ''}.png")


def statistical_tests(df: pd.DataFrame, reference: str = "phosopt_per_target") -> pd.DataFrame:
    """Wilcoxon signed-rank test of each method vs the reference."""
    if reference not in df["method"].unique():
        # Fallback: try 'phosopt'
        if "phosopt" in df["method"].unique():
            reference = "phosopt"
        else:
            print(f"  Reference method '{reference}' not found. Skipping stats.")
            return pd.DataFrame()

    ref_scores = df[df["method"] == reference].set_index("target_id")["score"]
    methods = [m for m in df["method"].unique() if m != reference]

    rows: list[dict] = []
    for method in methods:
        m_scores = df[df["method"] == method].set_index("target_id")["score"]
        common = ref_scores.index.intersection(m_scores.index)
        if len(common) < 3:
            continue
        try:
            stat, p = stats.wilcoxon(ref_scores.loc[common], m_scores.loc[common])
        except ValueError:
            stat, p = float("nan"), float("nan")
        rows.append({
            "method": method,
            "vs": reference,
            "n_pairs": len(common),
            "wilcoxon_stat": stat,
            "p_value": p,
            "ref_mean": ref_scores.loc[common].mean(),
            "method_mean": m_scores.loc[common].mean(),
        })
    return pd.DataFrame(rows).round(6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse PhosOpt experiment results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Filter by benchmark type (per_target, generalized, adaptation)")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    out = args.output_dir or args.results_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    df = load_logs(args.results_dir)
    print(f"Loaded {len(df)} result entries from {args.results_dir}")

    benchmarks = [args.benchmark] if args.benchmark else df["benchmark_type"].unique().tolist()

    for bm in benchmarks:
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {bm}")
        print(f"{'=' * 60}")

        table = summary_table(df, bm)
        print("\nSummary:")
        print(table.to_string())
        table.to_csv(out / f"summary_{bm}.csv")

        plot_scores(df, out, bm)
        plot_efficiency(df, out, bm)

        stat_df = statistical_tests(df[df["benchmark_type"] == bm])
        if not stat_df.empty:
            print("\nStatistical tests:")
            print(stat_df.to_string(index=False))
            stat_df.to_csv(out / f"stats_{bm}.csv", index=False)

    print(f"\nAnalysis output saved to {out}")


if __name__ == "__main__":
    main()
