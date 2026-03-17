"""
Create a single-page visual report from single_subject_report.json.

Outputs one PNG image summarizing:
  - learned implant params (alpha, beta, offset, shank_length)
  - training/validation curves
  - test metrics and baseline comparisons
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _project_root() -> Path:
    # code/visualize/report.py -> project root is two levels up
    return Path(__file__).resolve().parents[2]


def _safe_get(d: dict[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d else default


def _as_float_list(x: Any) -> list[float]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out: list[float] = []
        for v in x:
            try:
                out.append(float(v))
            except Exception:
                pass
        return out
    return []


def _format_metrics_row(metrics: dict[str, Any]) -> str:
    keys = ["mse", "dice", "hellinger"]
    parts = []
    for k in keys:
        if k in metrics:
            v = metrics[k]
            try:
                parts.append(f"{k}={float(v):.4g}")
            except Exception:
                parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "(no metrics)"


def make_report_figure(report: dict[str, Any]) -> plt.Figure:
    subject_id = str(_safe_get(report, "subject_id", "unknown_subject"))
    learned = _safe_get(report, "learned_implant_params", {}) or {}
    history = _safe_get(report, "history", {}) or {}
    test_metrics = _safe_get(report, "test_metrics", {}) or {}
    baselines = _safe_get(report, "baselines", {}) or {}

    train_total = _as_float_list(history.get("train_total"))
    train_recon = _as_float_list(history.get("train_recon"))
    val_mse = _as_float_list(history.get("val_mse"))
    val_dice = _as_float_list(history.get("val_dice"))
    val_hellinger = _as_float_list(history.get("val_hellinger"))

    n_epochs = max(
        len(train_total), len(train_recon), len(val_mse), len(val_dice), len(val_hellinger), 1
    )
    epochs = np.arange(1, n_epochs + 1)

    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.0], width_ratios=[1.3, 1.3, 1.0])

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_val = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[0, 2])
    ax_base = fig.add_subplot(gs[1, 0:2])
    ax_params = fig.add_subplot(gs[1, 2])

    fig.suptitle(f"PhosOpt training report: {subject_id}", fontsize=16, y=0.98)

    # --- (A) Train curves
    if train_total:
        ax_loss.plot(epochs[: len(train_total)], train_total, label="train_total")
    if train_recon:
        ax_loss.plot(epochs[: len(train_recon)], train_recon, label="train_recon (MSE)")
    ax_loss.set_title("Training curves")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="best", fontsize=9)

    # --- (B) Validation curves
    if val_mse:
        ax_val.plot(epochs[: len(val_mse)], val_mse, label="val_mse")
    if val_dice:
        ax_val.plot(epochs[: len(val_dice)], val_dice, label="val_dice")
    if val_hellinger:
        ax_val.plot(epochs[: len(val_hellinger)], val_hellinger, label="val_hellinger")
    ax_val.set_title("Validation curves")
    ax_val.set_xlabel("Epoch")
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(loc="best", fontsize=9)

    # --- (C) Text summary
    ax_text.axis("off")
    alpha = learned.get("alpha", None)
    beta = learned.get("beta", None)
    offset = learned.get("offset_from_base", None)
    shank = learned.get("shank_length", None)
    learned_lines = [
        "Learned implant params (shared):",
        f"  alpha: {alpha:.3f}" if isinstance(alpha, (int, float)) else f"  alpha: {alpha}",
        f"  beta: {beta:.3f}" if isinstance(beta, (int, float)) else f"  beta: {beta}",
        f"  offset_from_base: {offset:.3f}" if isinstance(offset, (int, float)) else f"  offset_from_base: {offset}",
        f"  shank_length: {shank:.3f}" if isinstance(shank, (int, float)) else f"  shank_length: {shank}",
        "",
        "Test metrics:",
        f"  {_format_metrics_row(test_metrics)}",
    ]
    if baselines:
        learned_lines.append("")
        learned_lines.append("Baselines:")
        for name, m in baselines.items():
            if isinstance(m, dict):
                learned_lines.append(f"  {name}: {_format_metrics_row(m)}")
            else:
                learned_lines.append(f"  {name}: {m}")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(learned_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    # --- (D) Baseline bar chart (MSE)
    ax_base.set_title("Test MSE comparison (lower is better)")
    labels = ["phosopt"] + list(baselines.keys())
    values = [float(test_metrics.get("mse", np.nan))]
    for k in baselines.keys():
        v = baselines[k]
        if isinstance(v, dict) and "mse" in v:
            values.append(float(v["mse"]))
        else:
            values.append(np.nan)
    x = np.arange(len(labels))
    ax_base.bar(x, values, color=["#1f77b4"] + ["#9aa0a6"] * (len(labels) - 1))
    ax_base.set_xticks(x, labels, rotation=20, ha="right")
    ax_base.set_ylabel("MSE")
    ax_base.grid(axis="y", alpha=0.3)
    for xi, yi in zip(x, values):
        if np.isfinite(yi):
            ax_base.text(xi, yi, f"{yi:.4g}", ha="center", va="bottom", fontsize=9)

    # --- (E) Params visualization
    ax_params.set_title("Shared params")
    p_names = ["alpha", "beta", "offset", "shank"]
    p_vals = [
        float(alpha) if isinstance(alpha, (int, float)) else np.nan,
        float(beta) if isinstance(beta, (int, float)) else np.nan,
        float(offset) if isinstance(offset, (int, float)) else np.nan,
        float(shank) if isinstance(shank, (int, float)) else np.nan,
    ]
    yy = np.arange(len(p_names))
    ax_params.barh(yy, p_vals, color="#2ca02c")
    ax_params.set_yticks(yy, p_names)
    ax_params.invert_yaxis()
    ax_params.grid(axis="x", alpha=0.3)
    for yi, v in zip(yy, p_vals):
        if np.isfinite(v):
            ax_params.text(v, yi, f" {v:.3g}", va="center", ha="left", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> None:
    root = _project_root()
    default_report = root / "data" / "output" / "inverse_training" / "single_subject_report.json"
    default_out_dir = root / "data" / "output"

    parser = argparse.ArgumentParser(description="Visualize single_subject_report.json as one image")
    parser.add_argument("--report", type=Path, default=default_report, help="Path to report json")
    parser.add_argument("--output-dir", "-o", type=Path, default=default_out_dir, help="Directory to save image")
    parser.add_argument("--output-name", type=str, default="single_subject_report.png", help="Output PNG filename")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    args = parser.parse_args()

    if not args.report.exists():
        raise FileNotFoundError(f"Not found: {args.report}")

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    fig = make_report_figure(report)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
