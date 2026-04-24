"""
Arena runner for PhosOpt-style strategy comparison.

For each target map:
1) Run phosopt once to get implant params and an active-electrode budget K.
2) Evaluate phosopt, allon, random, center, and intensity selection rules.
3) Save per-target artifacts under data/arena0422/<strategy>/<target>/.
4) Save strategy aggregates and a cross-strategy comparison bundle.

Notes:
- phosopt keeps its learned implant params fixed and determines the shared
  selection budget K from its active electrode count.
- allon uses all electrodes active.
- random, center, and intensity use the same K but different ordering rules.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

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


MODEL_PATH = PROJECT_ROOT / "data" / "cnnopt" / "baseline" / "train" / "single_subject_inverse_model.pt"
RETINOTOPY_DIR = PROJECT_ROOT / "data" / "fmri" / "100610"
HEMISPHERE = "LH"
MAP_SIZE = 128

TARGETS_DIR = Path(os.getenv("PHOSOPT_ARENA_TARGETS_DIR", str(PROJECT_ROOT / "data" / "targets0421")))
SAVE_ROOT = PROJECT_ROOT / "data" / "arena0422"
ELECTRODE_ON_THRESHOLD = 0.5

STRATEGIES = ("phosopt", "allon", "random", "center", "intensity")


class MetricsTracker:
    def __init__(self) -> None:
        self.simulator_calls = 0
        self.model_calls = 0
        self.forward_times: list[float] = []
        self.simulator_times: list[float] = []
        self.metric_times: list[float] = []


metrics_tracker = MetricsTracker()
_original_simulator_forward: Callable[..., Any] | None = None


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            probe = torch.zeros(1, device="cuda")
            _ = probe + probe
            return torch.device("cuda")
        except Exception:
            print("[WARNING] CUDA detected but GPU kernels failed. Falling back to CPU.")
    return torch.device("cpu")


def _stat(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_map_png(path: Path, arr: np.ndarray, title: str) -> None:
    plt.figure(figsize=(5, 5), dpi=150)
    plt.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def _print_progress(step: int, total_steps: int, message: str) -> None:
    pct = 100.0 * step / total_steps
    print(f"[PhosOpt][arena0422][{step}/{total_steps} | {pct:5.1f}%] {message}")


def _wrap_simulator(simulator: torch.nn.Module) -> None:
    global _original_simulator_forward
    _original_simulator_forward = simulator.forward

    def wrapped_forward(*args, **kwargs):
        if _original_simulator_forward is None:
            raise RuntimeError("Simulator forward wrapper not initialized")
        metrics_tracker.simulator_calls += 1
        start = time.time()
        result = _original_simulator_forward(*args, **kwargs)
        metrics_tracker.simulator_times.append(time.time() - start)
        return result

    simulator.forward = wrapped_forward


def _parse_target_map(target_path: Path) -> np.ndarray:
    target_np = np.load(target_path)
    if target_np.ndim == 3 and target_np.shape[0] == 1:
        target_np = target_np[0]
    elif target_np.ndim != 2:
        raise ValueError(f"Unexpected target shape for {target_path.name}: {target_np.shape}")

    if target_np.shape != (MAP_SIZE, MAP_SIZE):
        raise ValueError(f"Target {target_path.name} has shape {target_np.shape}, expected {(MAP_SIZE, MAP_SIZE)}")
    return target_np.astype(np.float32, copy=False)


def _safe_sample(image: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    h, w = image.shape
    row = np.clip(row, 0.0, h - 1.0)
    col = np.clip(col, 0.0, w - 1.0)

    r0 = np.floor(row).astype(np.int64)
    c0 = np.floor(col).astype(np.int64)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)

    dr = row - r0
    dc = col - c0

    top = image[r0, c0] * (1.0 - dc) + image[r0, c1] * dc
    bottom = image[r1, c0] * (1.0 - dc) + image[r1, c1] * dc
    return top * (1.0 - dr) + bottom * dr


def _theta_to_image_coords(prf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    angle_rad = np.deg2rad(prf[:, 0])
    ecc = prf[:, 1]
    x = ecc * np.cos(angle_rad)
    y = ecc * np.sin(angle_rad)
    scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1e-6)
    x_norm = x / scale
    y_norm = y / scale
    h, w = shape
    col = (x_norm + 1.0) * 0.5 * (w - 1)
    row = (1.0 - (y_norm + 1.0) * 0.5) * (h - 1)
    return np.stack([row, col], axis=1)


def _select_from_order(order: np.ndarray, k: int, total: int) -> np.ndarray:
    mask = np.zeros(total, dtype=bool)
    if k <= 0:
        return mask
    k = min(k, total)
    mask[order[:k]] = True
    return mask


def _mask_to_logits(mask: np.ndarray, device: torch.device, high: float = 12.0, low: float = -12.0) -> torch.Tensor:
    logits = torch.full((1, mask.size), low, dtype=torch.float32, device=device)
    if mask.any():
        logits[0, torch.from_numpy(mask.astype(np.bool_)).to(device)] = high
    return logits


def _build_order(strategy: str, target_np: np.ndarray, simulator: DifferentiableSimulator, params: torch.Tensor, base_probs: np.ndarray, seed: int) -> np.ndarray:
    total = base_probs.size
    if strategy == "phosopt":
        return np.argsort(-base_probs, kind="stable")
    if strategy == "allon":
        return np.arange(total, dtype=np.int64)
    if strategy == "random":
        return np.random.default_rng(seed).permutation(total)

    with torch.no_grad():
        alpha, beta, offset, shank = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        positioned = simulator._create_positioned_grid(alpha, beta, offset, shank)
        prf, validity = simulator._soft_prf_lookup(positioned)
    prf_np = prf[0].detach().cpu().numpy()
    validity_np = validity[0].detach().cpu().numpy()
    coords = _theta_to_image_coords(prf_np, target_np.shape)

    if strategy == "center":
        # Right-center prior: row = 1/2 height, col = 3/4 width.
        anchor = np.array([0.5 * (target_np.shape[0] - 1), 0.75 * (target_np.shape[1] - 1)], dtype=np.float32)
        score = -np.linalg.norm(coords - anchor[None, :], axis=1)
    elif strategy == "intensity":
        # Intensity prior: prefer electrodes mapped onto brighter target regions.
        sampled = _safe_sample(target_np, coords[:, 0], coords[:, 1])
        score = sampled
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    score = np.asarray(score, dtype=np.float64)
    score[validity_np <= 0.0] = -np.inf
    return np.argsort(-score, kind="stable")


def _evaluate_target_with_mask(
    *,
    strategy: str,
    target_name: str,
    target_path: Path,
    target_np: np.ndarray,
    target_tensor: torch.Tensor,
    model: InverseModel,
    simulator: DifferentiableSimulator,
    params: torch.Tensor,
    order: np.ndarray,
    selected_mask: np.ndarray,
    budget_k: int,
    device: torch.device,
    target_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before_sim_calls = metrics_tracker.simulator_calls
    before_model_calls = metrics_tracker.model_calls

    wall_start = time.time()
    with torch.no_grad():
        strategy_logits = _mask_to_logits(selected_mask, device=device)
        sim_start = time.time()
        recon = simulator(params, strategy_logits)
        sim_elapsed = time.time() - sim_start

        metric_start = time.time()
        dc_metric = float(dice_score(recon, target_tensor).item())
        hd_metric = float(hellinger_distance(recon, target_tensor).item())
        y_metric = float(y_metric_from_params(simulator, params).item())
        metric_elapsed = time.time() - metric_start
        metrics_tracker.metric_times.append(metric_elapsed)

    wall_elapsed = time.time() - wall_start

    recon_np = recon[0, 0].detach().cpu().numpy()
    selected_count = int(selected_mask.sum())
    active_ratio = float(selected_count / selected_mask.size)
    mean_activation = float(torch.sigmoid(strategy_logits)[0].detach().cpu().numpy().mean())
    score = dc_metric + 0.1 * y_metric - hd_metric
    loss = 2.0 - score

    model_calls_this = metrics_tracker.model_calls - before_model_calls
    sim_calls_this = metrics_tracker.simulator_calls - before_sim_calls

    params_vec = params[0].detach().cpu().numpy().tolist()
    learned_params = {
        "alpha": params_vec[0],
        "beta": params_vec[1],
        "offset_from_base": params_vec[2],
        "shank_length": params_vec[3],
    }

    result = {
        "strategy": strategy,
        "target_name": target_name,
        "target_source": str(target_path),
        "model_path": str(MODEL_PATH),
        "retinotopy_dir": str(RETINOTOPY_DIR),
        "hemisphere": HEMISPHERE,
        "map_size": MAP_SIZE,
        "electrode_on_threshold": ELECTRODE_ON_THRESHOLD,
        "budget_k": budget_k,
        "selected_count": selected_count,
        "selection_rule": strategy,
        "selected_indices_preview": [int(x) for x in order[: min(20, order.size)]],
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
            "active_electrode_count": selected_count,
            "active_ratio": active_ratio,
            "mean_activation": mean_activation,
        },
        "simulator_metrics": {
            "simulator_forward_calls": sim_calls_this,
            "model_forward_calls": model_calls_this,
            "simulator_forward_time_sec": sim_elapsed,
            "metric_computation_time_sec": metric_elapsed,
        },
        "timing": {
            "wall_clock_time_sec": wall_elapsed,
        },
    }

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(target_dir / "reconstruction.npy", recon_np)
    np.save(target_dir / "target.npy", target_np)
    np.save(target_dir / "selected_mask.npy", selected_mask.astype(np.uint8))
    np.save(target_dir / "selection_order.npy", order.astype(np.int32))
    np.save(target_dir / "electrode_activations.npy", torch.sigmoid(strategy_logits)[0].detach().cpu().numpy())

    _save_map_png(target_dir / "reconstruction.png", recon_np, f"{strategy}: {target_name}")
    _save_map_png(target_dir / "target.png", target_np, f"Target: {target_name}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    axes[0].imshow(target_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Target")
    axes[0].axis("off")
    axes[1].imshow(recon_np, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"{strategy} reconstruction")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(target_dir / "comparison.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    row = {
        "strategy": strategy,
        "target_name": target_name,
        "target_source": str(target_path),
        "budget_k": budget_k,
        "selected_count": selected_count,
        "dc_metric": dc_metric,
        "y_metric": y_metric,
        "hd_metric": hd_metric,
        "score": score,
        "loss": loss,
        "active_electrode_count": selected_count,
        "active_ratio": active_ratio,
        "mean_activation": mean_activation,
        "simulator_forward_calls": sim_calls_this,
        "model_forward_calls": model_calls_this,
        "wall_clock_time_sec": wall_elapsed,
        "simulator_forward_time_sec": sim_elapsed,
        "metric_computation_time_sec": metric_elapsed,
        "result_dir": str(target_dir),
    }
    return result, row


def _load_existing_row(results_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {
        "strategy": data.get("strategy"),
        "target_name": data.get("target_name"),
        "target_source": data.get("target_source"),
        "budget_k": int(data.get("budget_k", 0)),
        "selected_count": int(data.get("selected_count", data.get("efficiency_metrics", {}).get("active_electrode_count", 0))),
        "dc_metric": float(data.get("reconstruction_metrics", {}).get("dc_metric", 0.0)),
        "y_metric": float(data.get("reconstruction_metrics", {}).get("y_metric", 0.0)),
        "hd_metric": float(data.get("reconstruction_metrics", {}).get("hd_metric", 0.0)),
        "score": float(data.get("composite_scores", {}).get("score", 0.0)),
        "loss": float(data.get("composite_scores", {}).get("loss", 0.0)),
        "active_electrode_count": int(data.get("efficiency_metrics", {}).get("active_electrode_count", 0)),
        "active_ratio": float(data.get("efficiency_metrics", {}).get("active_ratio", 0.0)),
        "mean_activation": float(data.get("efficiency_metrics", {}).get("mean_activation", 0.0)),
        "simulator_forward_calls": int(data.get("simulator_metrics", {}).get("simulator_forward_calls", 0)),
        "model_forward_calls": int(data.get("simulator_metrics", {}).get("model_forward_calls", 0)),
        "wall_clock_time_sec": float(data.get("timing", {}).get("wall_clock_time_sec", 0.0)),
        "simulator_forward_time_sec": float(data.get("simulator_metrics", {}).get("simulator_forward_time_sec", 0.0)),
        "metric_computation_time_sec": float(data.get("simulator_metrics", {}).get("metric_computation_time_sec", 0.0)),
        "result_dir": str(results_path.parent),
    }


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_strategy_plots(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return

    target_names = [str(r["target_name"]) for r in rows]
    losses = np.asarray([float(r["loss"]) for r in rows], dtype=np.float64)
    scores = np.asarray([float(r["score"]) for r in rows], dtype=np.float64)
    dcs = np.asarray([float(r["dc_metric"]) for r in rows], dtype=np.float64)
    hds = np.asarray([float(r["hd_metric"]) for r in rows], dtype=np.float64)
    counts = np.asarray([float(r["active_electrode_count"]) for r in rows], dtype=np.float64)
    times = np.asarray([float(r["wall_clock_time_sec"]) for r in rows], dtype=np.float64)
    sim_times = np.asarray([float(r["simulator_forward_time_sec"]) for r in rows], dtype=np.float64)
    metric_times = np.asarray([float(r["metric_computation_time_sec"]) for r in rows], dtype=np.float64)

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

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
    metric_data = [
        (dcs, "DC"),
        (hds, "HD"),
        (scores, "Score"),
        (losses, "Loss"),
        (counts, "Active Electrodes"),
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

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    ax.scatter(counts, losses, s=20, alpha=0.75)
    for name, x_val, y_val in zip(target_names, counts, losses):
        ax.annotate(name, (x_val, y_val), textcoords="offset points", xytext=(4, 3), fontsize=7, alpha=0.85)
    if len(counts) >= 2 and np.std(counts) > 1e-12:
        coef = np.polyfit(counts, losses, deg=1)
        xline = np.linspace(float(counts.min()), float(counts.max()), 200)
        yline = coef[0] * xline + coef[1]
        ax.plot(xline, yline, linewidth=1.3)
    ax.set_xlabel("Active Electrode Count")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Active Electrode Count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_loss_vs_active_electrodes.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    order_time = np.argsort(times)[::-1]
    x_t = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(rows) * 0.38), 10), dpi=150, sharex=True)
    axes[0].plot(x_t, times[order_time], marker="o", linewidth=1.4, label="Wall")
    axes[0].plot(x_t, sim_times[order_time], marker="o", linewidth=1.2, label="Simulator")
    axes[0].plot(x_t, metric_times[order_time], marker="o", linewidth=1.2, label="Metric")
    axes[0].set_ylabel("Time (sec)")
    axes[0].set_title("Per-target Time Metrics (sorted by wall time)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")

    denom = np.clip(times[order_time], 1e-12, None)
    axes[1].plot(x_t, sim_times[order_time] / denom, marker="o", linewidth=1.2, label="Simulator / Wall")
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


def _save_comparison_plots(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return

    strategies = sorted({str(r["strategy"]) for r in rows})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["strategy"])].append(row)

    means = {s: {
        "score": np.mean([float(r["score"]) for r in grouped[s]]),
        "loss": np.mean([float(r["loss"]) for r in grouped[s]]),
        "count": np.mean([float(r["active_electrode_count"]) for r in grouped[s]]),
        "wall": np.mean([float(r["wall_clock_time_sec"]) for r in grouped[s]]),
    } for s in strategies}

    x = np.arange(len(strategies))
    labels = strategies

    def _bar(metric_key: str, title: str, filename: str) -> None:
        values = [means[s][metric_key] for s in labels]
        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / filename, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    _bar("score", "Mean Score by Strategy", "plot_strategy_mean_score.png")
    _bar("loss", "Mean Loss by Strategy", "plot_strategy_mean_loss.png")
    _bar("count", "Mean Active Electrode Count by Strategy", "plot_strategy_mean_active_count.png")
    _bar("wall", "Mean Wall Time by Strategy", "plot_strategy_mean_wall_time.png")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    data = [[float(r["score"]) for r in grouped[s]] for s in labels]
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_title("Score Distribution by Strategy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_strategy_score_boxplot.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _write_strategy_summary(rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    summary = {
        "num_rows": len(rows),
        "dc_metric": _stat([float(r["dc_metric"]) for r in rows]),
        "y_metric": _stat([float(r["y_metric"]) for r in rows]),
        "hd_metric": _stat([float(r["hd_metric"]) for r in rows]),
        "score": _stat([float(r["score"]) for r in rows]),
        "loss": _stat([float(r["loss"]) for r in rows]),
        "active_electrode_count": _stat([float(r["active_electrode_count"]) for r in rows]),
        "wall_clock_time_sec": _stat([float(r["wall_clock_time_sec"]) for r in rows]),
        "simulator_forward_calls": _stat([float(r["simulator_forward_calls"]) for r in rows]),
        "simulator_forward_time_sec": _stat([float(r["simulator_forward_time_sec"]) for r in rows]),
        "metric_computation_time_sec": _stat([float(r["metric_computation_time_sec"]) for r in rows]),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    device = _get_device()
    print(f"[PhosOpt][arena0422] Using device: {device}")

    if not TARGETS_DIR.exists():
        raise FileNotFoundError(f"Targets directory not found: {TARGETS_DIR}")
    target_files = sorted(TARGETS_DIR.glob("*.npy"))
    if not target_files:
        raise RuntimeError(f"No .npy target files found in: {TARGETS_DIR}")
    print(f"[PhosOpt][arena0422] Found {len(target_files)} targets in {TARGETS_DIR}")

    _ensure_dir(SAVE_ROOT)
    comparison_dir = SAVE_ROOT / "comparison"
    _ensure_dir(comparison_dir)

    _print_progress(1, 8, "Loading inverse model")
    model = InverseModel(in_channels=1, latent_dim=256, electrode_dim=1000, map_size=MAP_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    _print_progress(2, 8, "Initializing simulator")
    simulator = DifferentiableSimulator(data_dir=RETINOTOPY_DIR, hemisphere=HEMISPHERE, map_size=MAP_SIZE).to(device).eval()
    _wrap_simulator(simulator)

    strategy_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATEGIES}
    comparison_rows: list[dict[str, Any]] = []
    run_start = time.time()

    iterator = tqdm(target_files, desc="arena0422", unit="target") if tqdm else target_files
    for idx, target_path in enumerate(iterator, start=1):
        if tqdm is None:
            print(f"[PhosOpt][arena0422] target {idx}/{len(target_files)}: {target_path.name}")

        target_name = target_path.stem
        target_np = _parse_target_map(target_path)
        target_tensor = torch.from_numpy(target_np).unsqueeze(0).unsqueeze(0).to(device)

        phosopt_dir = SAVE_ROOT / "phosopt" / target_name
        phosopt_results = phosopt_dir / "results.json"

        if phosopt_results.exists():
            cached = _load_existing_row(phosopt_results)
            if cached is not None:
                phosopt_row = cached
                phosopt_budget = int(phosopt_row["selected_count"])
                phosopt_params = None
                phosopt_probs = None
                try:
                    data = json.loads(phosopt_results.read_text(encoding="utf-8"))
                    phosopt_params = torch.tensor(
                        [
                            [
                                data["learned_implant_params"]["alpha"],
                                data["learned_implant_params"]["beta"],
                                data["learned_implant_params"]["offset_from_base"],
                                data["learned_implant_params"]["shank_length"],
                            ]
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    phosopt_probs = np.load(phosopt_dir / "electrode_activations.npy")
                except Exception:
                    phosopt_results.unlink(missing_ok=True)
                    phosopt_budget = -1
                    phosopt_params = None
                    phosopt_probs = None
            else:
                phosopt_budget = -1
                phosopt_params = None
                phosopt_probs = None
        else:
            phosopt_budget = -1
            phosopt_params = None
            phosopt_probs = None

        if phosopt_budget < 0 or phosopt_params is None or phosopt_probs is None:
            with torch.no_grad():
                model_start = time.time()
                phosopt_params, phosopt_logits = model(target_tensor)
                metrics_tracker.model_calls += 1
                metrics_tracker.forward_times.append(time.time() - model_start)

                sim_start = time.time()
                phosopt_recon = simulator(phosopt_params, phosopt_logits)
                phosopt_sim_time = time.time() - sim_start

                metric_start = time.time()
                phosopt_dc = float(dice_score(phosopt_recon, target_tensor).item())
                phosopt_hd = float(hellinger_distance(phosopt_recon, target_tensor).item())
                phosopt_y = float(y_metric_from_params(simulator, phosopt_params).item())
                phosopt_metric_time = time.time() - metric_start
                metrics_tracker.metric_times.append(phosopt_metric_time)

            phosopt_probs = torch.sigmoid(phosopt_logits)[0].detach().cpu().numpy()
            phosopt_mask = phosopt_probs >= ELECTRODE_ON_THRESHOLD
            if phosopt_mask.sum() == 0:
                phosopt_mask[int(np.argmax(phosopt_probs))] = True
            phosopt_budget = int(phosopt_mask.sum())

            phosopt_order = np.argsort(-phosopt_probs, kind="stable")
            phosopt_dir.mkdir(parents=True, exist_ok=True)
            phosopt_result, phosopt_row = _evaluate_target_with_mask(
                strategy="phosopt",
                target_name=target_name,
                target_path=target_path,
                target_np=target_np,
                target_tensor=target_tensor,
                model=model,
                simulator=simulator,
                params=phosopt_params,
                order=phosopt_order,
                selected_mask=phosopt_mask,
                budget_k=phosopt_budget,
                device=device,
                target_dir=phosopt_dir,
            )
            strategy_rows["phosopt"].append(phosopt_row)
            comparison_rows.append(phosopt_row)

        else:
            # If the cached phosopt result exists, reuse its stored budget and params.
            data = json.loads(phosopt_results.read_text(encoding="utf-8"))
            phosopt_params = torch.tensor(
                [[
                    data["learned_implant_params"]["alpha"],
                    data["learned_implant_params"]["beta"],
                    data["learned_implant_params"]["offset_from_base"],
                    data["learned_implant_params"]["shank_length"],
                ]],
                dtype=torch.float32,
                device=device,
            )
            phosopt_probs = np.load(phosopt_dir / "electrode_activations.npy")
            phosopt_order = np.argsort(-phosopt_probs, kind="stable")
            phosopt_mask = np.load(phosopt_dir / "selected_mask.npy").astype(bool)
            phosopt_row = _load_existing_row(phosopt_results)
            if phosopt_row is None:
                raise RuntimeError(f"Unable to load cached phosopt result: {phosopt_results}")
            strategy_rows["phosopt"].append(phosopt_row)
            comparison_rows.append(phosopt_row)

        assert phosopt_params is not None
        assert phosopt_probs is not None

        _print_progress(3, 8, f"Evaluating strategies for {target_name}")
        for strategy in ("allon", "random", "center", "intensity"):
            strategy_dir = SAVE_ROOT / strategy / target_name
            results_path = strategy_dir / "results.json"
            if results_path.exists():
                cached = _load_existing_row(results_path)
                if cached is not None:
                    strategy_rows[strategy].append(cached)
                    comparison_rows.append(cached)
                    continue

            if strategy == "allon":
                order = np.arange(phosopt_probs.size, dtype=np.int64)
                selected_mask = np.ones_like(phosopt_probs, dtype=bool)
            else:
                order = _build_order(strategy, target_np, simulator, phosopt_params, phosopt_probs, seed=abs(hash((strategy, target_name))) % (2**32))
                selected_mask = _select_from_order(order, phosopt_budget, phosopt_probs.size)

            result, row = _evaluate_target_with_mask(
                strategy=strategy,
                target_name=target_name,
                target_path=target_path,
                target_np=target_np,
                target_tensor=target_tensor,
                model=model,
                simulator=simulator,
                params=phosopt_params,
                order=order,
                selected_mask=selected_mask,
                budget_k=phosopt_budget,
                device=device,
                target_dir=strategy_dir,
            )
            strategy_rows[strategy].append(row)
            comparison_rows.append(row)

    run_elapsed = time.time() - run_start

    _print_progress(4, 8, "Writing strategy aggregates")
    strategy_summaries: dict[str, dict[str, Any]] = {}
    for strategy in STRATEGIES:
        rows = strategy_rows[strategy]
        strategy_root = SAVE_ROOT / strategy
        aggregate_dir = strategy_root / "aggregate"
        _ensure_dir(aggregate_dir)

        summary = _write_strategy_summary(rows, aggregate_dir / "summary.json")
        strategy_summaries[strategy] = summary
        _write_jsonl(rows, aggregate_dir / "per_target_metrics.jsonl")
        _write_table(rows, aggregate_dir / "per_target_metrics.csv")

        sorted_by_loss = sorted(rows, key=lambda r: float(r["loss"]))
        rank_lines = ["Best targets by loss:"]
        for i, row in enumerate(sorted_by_loss[:10], start=1):
            rank_lines.append(f"{i}. {row['target_name']} | loss={row['loss']:.6f} | score={row['score']:.6f}")
        rank_lines.append("")
        rank_lines.append("Worst targets by loss:")
        for i, row in enumerate(sorted_by_loss[-10:], start=1):
            rank_lines.append(f"{i}. {row['target_name']} | loss={row['loss']:.6f} | score={row['score']:.6f}")
        (aggregate_dir / "ranking_by_loss.txt").write_text("\n".join(rank_lines), encoding="utf-8")
        _save_strategy_plots(rows, aggregate_dir)

    _print_progress(5, 8, "Writing comparison outputs")
    _write_jsonl(comparison_rows, comparison_dir / "per_target_metrics.jsonl")
    _write_table(comparison_rows, comparison_dir / "per_target_metrics.csv")
    comparison_summary = {
        "targets_dir": str(TARGETS_DIR),
        "num_targets": len(target_files),
        "strategies": STRATEGIES,
        "run_elapsed_sec": run_elapsed,
        "strategy_summaries": strategy_summaries,
    }
    (comparison_dir / "summary.json").write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")
    _save_comparison_plots(comparison_rows, comparison_dir)

    _print_progress(6, 8, "Printing summary")
    print("\n" + "=" * 80)
    print("ARENA SUMMARY")
    print("=" * 80)
    print(f"Targets dir: {TARGETS_DIR}")
    print(f"Num targets: {len(target_files)}")
    print(f"Output root: {SAVE_ROOT}")
    print(f"Total wall-clock time: {run_elapsed:.6f} sec")
    print()
    for strategy in STRATEGIES:
        s = strategy_summaries[strategy]
        print(f"[{strategy}]")
        print(f"  score mean: {s['score']['mean']:.6f}")
        print(f"  loss mean: {s['loss']['mean']:.6f}")
        print(f"  active count mean: {s['active_electrode_count']['mean']:.2f}")
        print(f"  wall time mean: {s['wall_clock_time_sec']['mean']:.6f} sec")
    print("=" * 80)


if __name__ == "__main__":
    main()