# D:\yongtae\phosopt\code\infer0421.py

"""
Inference script for PhosOpt inverse model on single target.

Given:
- a trained inverse model checkpoint (.pt),
- a single target phosphene map (.npy),
- and the retinotopy directory for the simulator,

this script:
1) Loads the trained InverseModel and DifferentiableSimulator.
2) Runs inference on the single target.
3) Saves comprehensive metrics: reconstruction, efficiency, and timing.
4) Generates analysis summary with detailed statistics.
"""

from __future__ import annotations

import json
import os
import sys
import time
import csv
from pathlib import Path
from typing import Any, Callable

# Work around duplicate OpenMP runtime initialization on some Windows setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from loss.losses import dice_score, hellinger_distance, y_metric_from_params  # noqa: E402
from models.inverse_model import InverseModel  # noqa: E402
from simulator.physics_forward_torch import DifferentiableSimulator  # noqa: E402


# -----------------------------------------------------------------------------
# Inference configuration
# -----------------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "data" / "cnnopt" / "baseline" / "train" / "single_subject_inverse_model.pt"
RETINOTOPY_DIR = PROJECT_ROOT / "data" / "fmri" / "100610"
HEMISPHERE = "LH"
MAP_SIZE = 128

TARGET_FILE = PROJECT_ROOT / "data" / "targets0421" / "arc_00.npy"
TARGETS_DIR = PROJECT_ROOT / "data" / "targets0421"
SAVE_ROOT = PROJECT_ROOT / "data" / "infer0421"
ELECTRODE_ON_THRESHOLD = 0.5
AGGREGATE_DIRNAME = "aggregate"


# Metrics tracking
class MetricsTracker:
    """Track simulator calls and timing statistics."""
    
    def __init__(self):
        self.simulator_calls = 0
        self.model_calls = 0
        self.forward_times: list[float] = []
        self.simulator_times: list[float] = []
        self.metric_times: list[float] = []


metrics_tracker = MetricsTracker()

# Original simulator forward method
_original_simulator_forward: Callable[..., Any] | None = None


def _wrap_simulator(simulator: torch.nn.Module) -> None:
    """Wrap simulator forward to track calls and timing."""
    global _original_simulator_forward
    _original_simulator_forward = simulator.forward
    
    def wrapped_forward(*args, **kwargs):
        if _original_simulator_forward is None:
            raise RuntimeError("Simulator forward wrapper not initialized")
        metrics_tracker.simulator_calls += 1
        start = time.time()
        result = _original_simulator_forward(*args, **kwargs)
        elapsed = time.time() - start
        metrics_tracker.simulator_times.append(elapsed)
        return result
    
    simulator.forward = wrapped_forward


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda")
            _ = t + t
            return torch.device("cuda")
        except Exception:
            print("[WARNING] CUDA detected but GPU kernels failed. Falling back to CPU.")
    return torch.device("cpu")


def _stat(values: list[float]) -> dict[str, float]:
    """Compute statistics for a list of values."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _save_map_png(path: Path, arr: np.ndarray, title: str) -> None:
    """Save a map as PNG visualization."""
    plt.figure(figsize=(5, 5), dpi=150)
    plt.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def _print_progress(step: int, total_steps: int, message: str) -> None:
    pct = (step / total_steps) * 100.0
    print(f"[PhosOpt][infer0421][{step}/{total_steps} | {pct:5.1f}%] {message}")


def _row_from_results_json(results_path: Path, target_path: Path, target_dir: Path) -> dict[str, float | int | str]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    recon = data.get("reconstruction_metrics", {})
    comp = data.get("composite_scores", {})
    eff = data.get("efficiency_metrics", {})
    sim = data.get("simulator_metrics", {})
    timing = data.get("timing", {})
    return {
        "target_name": target_path.stem,
        "target_source": str(target_path),
        "dc_metric": float(recon["dc_metric"]),
        "y_metric": float(recon["y_metric"]),
        "hd_metric": float(recon["hd_metric"]),
        "score": float(comp["score"]),
        "loss": float(comp["loss"]),
        "active_electrode_count": int(eff["active_electrode_count"]),
        "active_ratio": float(eff["active_ratio"]),
        "mean_activation": float(eff["mean_activation"]),
        "simulator_forward_calls": int(sim.get("simulator_forward_calls", 0)),
        "model_forward_calls": int(sim.get("model_forward_calls", 0)),
        "wall_clock_time_sec": float(timing.get("wall_clock_time_sec", 0.0)),
        "model_forward_time_sec": float(sim.get("model_forward_time_sec", 0.0)),
        "simulator_forward_time_sec": float(sim.get("simulator_forward_time_sec", 0.0)),
        "metric_computation_time_sec": float(sim.get("metric_computation_time_sec", 0.0)),
        "result_dir": str(target_dir),
    }


def _save_aggregate_plots(rows: list[dict[str, float | int | str]], out_dir: Path) -> None:
    if not rows:
        return

    target_names = [str(r["target_name"]) for r in rows]
    losses = np.asarray([float(r["loss"]) for r in rows], dtype=np.float64)
    scores = np.asarray([float(r["score"]) for r in rows], dtype=np.float64)
    dcs = np.asarray([float(r["dc_metric"]) for r in rows], dtype=np.float64)
    hds = np.asarray([float(r["hd_metric"]) for r in rows], dtype=np.float64)
    actives = np.asarray([float(r["active_electrode_count"]) for r in rows], dtype=np.float64)
    times = np.asarray([float(r["wall_clock_time_sec"]) for r in rows], dtype=np.float64)
    model_times = np.asarray([float(r["model_forward_time_sec"]) for r in rows], dtype=np.float64)
    simulator_times = np.asarray([float(r["simulator_forward_time_sec"]) for r in rows], dtype=np.float64)
    metric_times = np.asarray([float(r["metric_computation_time_sec"]) for r in rows], dtype=np.float64)

    # 1) Loss/Score bar chart by target
    order = np.argsort(losses)
    x = np.arange(len(rows))
    fig, ax1 = plt.subplots(figsize=(max(10, len(rows) * 0.35), 5.5), dpi=150)
    ax1.bar(x, losses[order], alpha=0.75, label="Loss")
    ax1.set_xlabel("Target (sorted by loss)")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(x)
    ax1.set_xticklabels([target_names[i] for i in order], rotation=75, ha="right", fontsize=8)
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, scores[order], linewidth=1.5, label="Score")
    ax2.set_ylabel("Score")
    ax1.set_title("Per-target Loss / Score")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_loss_score_by_target.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # 2) Metric distributions
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
    metric_data = [
        (dcs, "DC"),
        (hds, "HD"),
        (scores, "Score"),
        (losses, "Loss"),
        (actives, "Active Electrodes"),
        (times, "Wall Time (s)"),
    ]
    for ax, (vals, title) in zip(axes.flat, metric_data):
        ax.hist(vals, bins=min(20, max(5, len(vals) // 2)))
        ax.axvline(float(vals.mean()), linestyle="--", linewidth=1.0)
        ax.set_title(title)
    fig.suptitle("Aggregate Metric Distributions", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_dir / "plot_metric_distributions.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # 3) Active electrodes vs loss scatter
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    ax.scatter(actives, losses, s=20, alpha=0.75)
    for name, x_val, y_val in zip(target_names, actives, losses):
        ax.annotate(
            name,
            (x_val, y_val),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=7,
            alpha=0.85,
        )
    if len(actives) >= 2 and np.std(actives) > 1e-12:
        coef = np.polyfit(actives, losses, deg=1)
        xline = np.linspace(float(actives.min()), float(actives.max()), 200)
        yline = coef[0] * xline + coef[1]
        ax.plot(xline, yline, linewidth=1.3)
    ax.set_xlabel("Active Electrode Count")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Active Electrode Count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_loss_vs_active_electrodes.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # 4) Per-target time metrics comparison
    order_time = np.argsort(times)[::-1]
    x_t = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(rows) * 0.38), 10), dpi=150, sharex=True)

    # Absolute time per target
    axes[0].plot(x_t, times[order_time], marker="o", linewidth=1.4, label="Wall")
    axes[0].plot(x_t, model_times[order_time], marker="o", linewidth=1.2, label="Model")
    axes[0].plot(x_t, simulator_times[order_time], marker="o", linewidth=1.2, label="Simulator")
    axes[0].plot(x_t, metric_times[order_time], marker="o", linewidth=1.2, label="Metric")
    axes[0].set_ylabel("Time (sec)")
    axes[0].set_title("Per-target Time Metrics (sorted by wall time)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")

    # Time composition ratio per target
    denom = np.clip(times[order_time], 1e-12, None)
    axes[1].plot(x_t, model_times[order_time] / denom, marker="o", linewidth=1.2, label="Model / Wall")
    axes[1].plot(x_t, simulator_times[order_time] / denom, marker="o", linewidth=1.2, label="Simulator / Wall")
    axes[1].plot(x_t, metric_times[order_time] / denom, marker="o", linewidth=1.2, label="Metric / Wall")
    axes[1].set_ylabel("Ratio")
    axes[1].set_xlabel("Target (sorted by wall time)")
    axes[1].set_xticks(x_t)
    axes[1].set_xticklabels([target_names[i] for i in order_time], rotation=75, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "plot_time_metrics_by_target.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    global metrics_tracker
    total_steps = 8
    
    _print_progress(1, total_steps, "Resolving device")
    device = _get_device()
    print(f"[PhosOpt][infer0421] Using device: {device}")
    
    # ------------------------------------------------------------------
    # Load target
    # ------------------------------------------------------------------
    _print_progress(2, total_steps, "Discovering target maps")
    if not TARGETS_DIR.exists():
        raise FileNotFoundError(f"Targets directory not found: {TARGETS_DIR}")
    target_files = sorted(TARGETS_DIR.glob("*.npy"))
    if not target_files:
        raise RuntimeError(f"No .npy target files found in: {TARGETS_DIR}")
    print(f"[PhosOpt][infer0421] Found {len(target_files)} target maps in: {TARGETS_DIR}")
    
    # ------------------------------------------------------------------
    # Load model and simulator
    # ------------------------------------------------------------------
    _print_progress(3, total_steps, "Loading inverse model checkpoint")
    print(f"[PhosOpt][infer0421] Loading model from: {MODEL_PATH}")
    model = InverseModel(in_channels=1, latent_dim=256, electrode_dim=1000, map_size=MAP_SIZE)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    
    _print_progress(4, total_steps, "Initializing differentiable simulator")
    print(f"[PhosOpt][infer0421] Initializing DifferentiableSimulator from {RETINOTOPY_DIR}")
    simulator = DifferentiableSimulator(
        data_dir=RETINOTOPY_DIR,
        hemisphere=HEMISPHERE,
        map_size=MAP_SIZE,
    ).to(device).eval()
    
    # Wrap simulator to track calls and timing
    _wrap_simulator(simulator)
    
    _print_progress(5, total_steps, "Running inference for all targets")
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    aggregate_dir = SAVE_ROOT / AGGREGATE_DIRNAME
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    all_dc: list[float] = []
    all_y: list[float] = []
    all_hd: list[float] = []
    all_score: list[float] = []
    all_loss: list[float] = []
    all_active_count: list[float] = []
    all_active_ratio: list[float] = []
    all_mean_activation: list[float] = []
    all_wall_time: list[float] = []
    per_target_rows: list[dict[str, float | int | str]] = []
    skipped_targets = 0
    processed_targets = 0

    run_start = time.time()
    iterator = tqdm(target_files, desc="infer0421", unit="target") if tqdm else target_files

    for i, target_path in enumerate(iterator, start=1):
        if tqdm is None:
            print(f"[PhosOpt][infer0421] target {i}/{len(target_files)}: {target_path.name}")

        target_name = target_path.stem
        target_dir = SAVE_ROOT / target_name
        results_path = target_dir / "results.json"

        # Reuse existing target results to skip re-inference.
        if results_path.exists():
            try:
                row = _row_from_results_json(results_path, target_path, target_dir)
                per_target_rows.append(row)
                all_dc.append(float(row["dc_metric"]))
                all_y.append(float(row["y_metric"]))
                all_hd.append(float(row["hd_metric"]))
                all_score.append(float(row["score"]))
                all_loss.append(float(row["loss"]))
                all_active_count.append(float(row["active_electrode_count"]))
                all_active_ratio.append(float(row["active_ratio"]))
                all_mean_activation.append(float(row["mean_activation"]))
                all_wall_time.append(float(row["wall_clock_time_sec"]))
                skipped_targets += 1
                if tqdm is None:
                    print(f"[PhosOpt][infer0421] skipped existing: {target_name}")
                continue
            except Exception as e:
                print(f"[PhosOpt][infer0421][WARN] Failed to reuse {results_path}: {e}. Recomputing.")

        target_np = np.load(target_path)
        if target_np.ndim == 3 and target_np.shape[0] == 1:
            target_np = target_np[0]
        elif target_np.ndim != 2:
            raise ValueError(f"Unexpected target shape for {target_path.name}: {target_np.shape}")

        if target_np.shape != (MAP_SIZE, MAP_SIZE):
            raise ValueError(
                f"Target {target_path.name} has shape {target_np.shape}, expected {(MAP_SIZE, MAP_SIZE)}"
            )

        target = torch.from_numpy(target_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        before_sim_calls = metrics_tracker.simulator_calls
        before_model_calls = metrics_tracker.model_calls
        wall_start = time.time()

        with torch.no_grad():
            model_start = time.time()
            params, electrode_logits = model(target)
            metrics_tracker.model_calls += 1
            model_elapsed = time.time() - model_start
            metrics_tracker.forward_times.append(model_elapsed)

            sim_start = time.time()
            recon = simulator(params, electrode_logits)
            sim_elapsed = time.time() - sim_start

            metric_start = time.time()
            dc_metric = float(dice_score(recon, target).item())
            hd_metric = float(hellinger_distance(recon, target).item())
            y_metric = float(y_metric_from_params(simulator, params).item())
            metric_elapsed = time.time() - metric_start
            metrics_tracker.metric_times.append(metric_elapsed)

        wall_elapsed = time.time() - wall_start

        params_vec = params[0].detach().cpu().numpy().tolist()
        elec_prob = torch.sigmoid(electrode_logits)[0].detach().cpu().numpy()
        recon_np = recon[0, 0].detach().cpu().numpy()

        num_active_electrodes = int((elec_prob > ELECTRODE_ON_THRESHOLD).sum())
        active_ratio = float(num_active_electrodes / elec_prob.size)
        score = dc_metric + 0.1 * y_metric - hd_metric
        loss = 2.0 - score

        target_dir.mkdir(parents=True, exist_ok=True)

        learned_params = {
            "alpha": params_vec[0],
            "beta": params_vec[1],
            "offset_from_base": params_vec[2],
            "shank_length": params_vec[3],
        }

        sim_calls_this_target = metrics_tracker.simulator_calls - before_sim_calls
        model_calls_this_target = metrics_tracker.model_calls - before_model_calls

        results = {
            "target_name": target_name,
            "target_source": str(target_path),
            "model_path": str(MODEL_PATH),
            "retinotopy_dir": str(RETINOTOPY_DIR),
            "hemisphere": HEMISPHERE,
            "map_size": MAP_SIZE,
            "electrode_on_threshold": ELECTRODE_ON_THRESHOLD,
            "learned_implant_params": learned_params,
            "reconstruction_metrics": {
                "dc_metric": dc_metric,
                "y_metric": y_metric,
                "hd_metric": hd_metric,
            },
            "composite_scores": {
                "score": score,
                "loss": loss,
            },
            "efficiency_metrics": {
                "active_electrode_count": num_active_electrodes,
                "active_ratio": active_ratio,
                "mean_activation": float(elec_prob.mean()),
                "max_activation": float(elec_prob.max()),
                "min_activation": float(elec_prob.min()),
            },
            "simulator_metrics": {
                "simulator_forward_calls": sim_calls_this_target,
                "model_forward_calls": model_calls_this_target,
                "simulator_forward_time_sec": sim_elapsed,
                "model_forward_time_sec": model_elapsed,
                "metric_computation_time_sec": metric_elapsed,
            },
            "timing": {
                "wall_clock_time_sec": wall_elapsed,
            },
        }

        (target_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        np.save(target_dir / "electrode_activations.npy", elec_prob)
        np.save(target_dir / "reconstruction.npy", recon_np)
        np.save(target_dir / "target.npy", target_np)

        _save_map_png(target_dir / "reconstruction.png", recon_np, f"Reconstruction: {target_name}")
        _save_map_png(target_dir / "target.png", target_np, f"Target: {target_name}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
        axes[0].imshow(target_np, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title("Target")
        axes[0].axis("off")
        axes[1].imshow(recon_np, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].set_title("Reconstruction")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(target_dir / "comparison.png", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

        row = {
            "target_name": target_name,
            "target_source": str(target_path),
            "dc_metric": dc_metric,
            "y_metric": y_metric,
            "hd_metric": hd_metric,
            "score": score,
            "loss": loss,
            "active_electrode_count": num_active_electrodes,
            "active_ratio": active_ratio,
            "mean_activation": float(elec_prob.mean()),
            "simulator_forward_calls": sim_calls_this_target,
            "model_forward_calls": model_calls_this_target,
            "wall_clock_time_sec": wall_elapsed,
            "model_forward_time_sec": model_elapsed,
            "simulator_forward_time_sec": sim_elapsed,
            "metric_computation_time_sec": metric_elapsed,
            "result_dir": str(target_dir),
        }
        per_target_rows.append(row)
        processed_targets += 1

        all_dc.append(dc_metric)
        all_y.append(y_metric)
        all_hd.append(hd_metric)
        all_score.append(score)
        all_loss.append(loss)
        all_active_count.append(float(num_active_electrodes))
        all_active_ratio.append(active_ratio)
        all_mean_activation.append(float(elec_prob.mean()))
        all_wall_time.append(wall_elapsed)

    run_elapsed = time.time() - run_start

    _print_progress(6, total_steps, "Saving aggregate summaries")
    summary = {
        "targets_dir": str(TARGETS_DIR),
        "num_targets": len(target_files),
        "model_path": str(MODEL_PATH),
        "retinotopy_dir": str(RETINOTOPY_DIR),
        "hemisphere": HEMISPHERE,
        "map_size": MAP_SIZE,
        "electrode_on_threshold": ELECTRODE_ON_THRESHOLD,
        "aggregate_reconstruction_metrics": {
            "dc_metric": _stat(all_dc),
            "y_metric": _stat(all_y),
            "hd_metric": _stat(all_hd),
        },
        "aggregate_composite_scores": {
            "score": _stat(all_score),
            "loss": _stat(all_loss),
        },
        "aggregate_efficiency_metrics": {
            "active_electrode_count": _stat(all_active_count),
            "active_ratio": _stat(all_active_ratio),
            "mean_activation": _stat(all_mean_activation),
        },
        "aggregate_runtime_metrics": {
            "per_target_wall_clock_time_sec": _stat(all_wall_time),
            "total_wall_clock_time_sec": run_elapsed,
            "total_simulator_forward_calls": metrics_tracker.simulator_calls,
            "total_model_forward_calls": metrics_tracker.model_calls,
            "total_simulator_time_sec": float(sum(metrics_tracker.simulator_times)),
            "total_model_forward_time_sec": float(sum(metrics_tracker.forward_times)),
            "total_metric_computation_time_sec": float(sum(metrics_tracker.metric_times)),
        },
        "run_counts": {
            "processed_targets": processed_targets,
            "skipped_existing_targets": skipped_targets,
        },
    }
    (aggregate_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (aggregate_dir / "per_target_metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in per_target_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_fields = [
        "target_name",
        "target_source",
        "dc_metric",
        "y_metric",
        "hd_metric",
        "score",
        "loss",
        "active_electrode_count",
        "active_ratio",
        "mean_activation",
        "simulator_forward_calls",
        "model_forward_calls",
        "wall_clock_time_sec",
        "model_forward_time_sec",
        "simulator_forward_time_sec",
        "metric_computation_time_sec",
        "result_dir",
    ]
    with (aggregate_dir / "per_target_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(per_target_rows)

    sorted_by_loss = sorted(per_target_rows, key=lambda r: float(r["loss"]))
    best10 = sorted_by_loss[:10]
    worst10 = sorted_by_loss[-10:]
    rank_lines = ["Best targets by loss:"]
    rank_lines.extend([f"{i+1}. {r['target_name']} | loss={r['loss']:.6f} | score={r['score']:.6f}" for i, r in enumerate(best10)])
    rank_lines.append("")
    rank_lines.append("Worst targets by loss:")
    rank_lines.extend([f"{i+1}. {r['target_name']} | loss={r['loss']:.6f} | score={r['score']:.6f}" for i, r in enumerate(worst10)])
    (aggregate_dir / "ranking_by_loss.txt").write_text("\n".join(rank_lines), encoding="utf-8")

    _save_aggregate_plots(per_target_rows, aggregate_dir)

    _print_progress(7, total_steps, "Printing aggregate report")
    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("INFERENCE SUMMARY (ALL TARGETS)")
    print("=" * 80)
    print(f"Targets dir: {TARGETS_DIR}")
    print(f"Num targets: {len(target_files)}")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Processed targets: {processed_targets}")
    print(f"Skipped existing targets: {skipped_targets}")
    print()
    print("RECONSTRUCTION METRICS (mean):")
    print(f"  DC (Dice Coefficient):        {summary['aggregate_reconstruction_metrics']['dc_metric']['mean']:.6f}")
    print(f"  Y (Yield):                    {summary['aggregate_reconstruction_metrics']['y_metric']['mean']:.6f}")
    print(f"  HD (Hellinger Distance):      {summary['aggregate_reconstruction_metrics']['hd_metric']['mean']:.6f}")
    print()
    print("COMPOSITE SCORES (mean):")
    print(f"  Score = DC + 0.1*Y - HD:      {summary['aggregate_composite_scores']['score']['mean']:.6f}")
    print(f"  Loss = 2 - Score:             {summary['aggregate_composite_scores']['loss']['mean']:.6f}")
    print()
    print("EFFICIENCY METRICS (mean):")
    print(f"  Active Electrode Count:       {summary['aggregate_efficiency_metrics']['active_electrode_count']['mean']:.2f}")
    print(f"  Active Ratio:                 {summary['aggregate_efficiency_metrics']['active_ratio']['mean']:.6f}")
    print(f"  Mean Activation:              {summary['aggregate_efficiency_metrics']['mean_activation']['mean']:.6f}")
    print()
    print("SIMULATOR / COST METRICS:")
    print(f"  Simulator Forward Calls:      {metrics_tracker.simulator_calls}")
    print(f"  Model Forward Calls:          {metrics_tracker.model_calls}")
    print(f"  Total Simulator Time:         {sum(metrics_tracker.simulator_times):.6f} sec")
    print(f"  Total Model Forward Time:     {sum(metrics_tracker.forward_times):.6f} sec")
    print(f"  Total Metric Computation:     {sum(metrics_tracker.metric_times):.6f} sec")
    print()
    print("TIMING:")
    print(f"  Total Wall-Clock Time:        {run_elapsed:.6f} sec")
    print(f"  Mean Per-Target Time:         {summary['aggregate_runtime_metrics']['per_target_wall_clock_time_sec']['mean']:.6f} sec")
    print(f"  Max  Per-Target Time:         {summary['aggregate_runtime_metrics']['per_target_wall_clock_time_sec']['max']:.6f} sec")
    print("=" * 80)
    print(f"All outputs saved to: {SAVE_ROOT}")
    print(f"Aggregate outputs: {aggregate_dir}")
    print("=" * 80)

    _print_progress(8, total_steps, "Done")


if __name__ == "__main__":
    main()
