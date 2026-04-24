"""
Batch pipeline for multi-subject, multi-hemisphere experiments.

Runs train -> infer -> analysis for all combinations of:
- subjects: 100610, 102311, 102816
- hemispheres: LH, RH

Outputs are stored as:
  data/0422_model/<subject>_<hemi>/
    train/
    infer/
    analysis/
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset import PhospheneDataset, load_letters_phosphene_splits  # noqa: E402
from loss.losses import LossConfig, dice_score, hellinger_distance, y_metric_from_params  # noqa: E402
from models.inverse_model import InverseModel  # noqa: E402
from simulator.physics_forward_torch_v2 import DifferentiableSimulatorIndependent  # noqa: E402
from trainer import (  # noqa: E402
    TrainConfig,
    evaluate_four_param_baseline,
    evaluate_inverse_model,
    evaluate_random_baseline,
    load_checkpoint,
    train_inverse_model,
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SUBJECT_IDS = ["100610", "102311", "102816"]
HEMISPHERES = ["LH", "RH"]

MAPS_NPZ = PROJECT_ROOT / "data" / "letters" / "emnist_letters_v3_halfright_128.npz"
VAL_RATIO_FROM_TRAIN = 0.1
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None

MAP_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3
SHARED_PARAMS_LR = 1e-2
MIN_EPOCHS = 10
EARLY_STOP_PATIENCE = 8
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
EARLY_STOP_MIN_DELTA = 1e-4
MONITOR_METRIC = "total_loss"
MONITOR_MODE = "min"
SEED = 42

OUTPUT_ROOT = PROJECT_ROOT / "data" / "0422_model"
SKIP_IF_EXISTS = True


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda")
            _ = t + t
            return torch.device("cuda")
        except Exception:
            print("[WARNING] CUDA detected but GPU kernels failed. Falling back to CPU.")
    return torch.device("cpu")


def _get_cpu_worker_count() -> int:
    total = os.cpu_count() or 1
    workers = int(total * 0.8)
    return max(1, workers)


def _stat(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def _save_hist_grid(columns: dict[str, np.ndarray], save_path: Path) -> None:
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
    metric_names = [k for k in keys if k in columns]
    if not metric_names:
        return

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
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(save_path)
    plt.close(fig)


def _save_scatter(rows: list[dict[str, Any]], save_path: Path) -> None:
    x_vals: list[float] = []
    y_vals: list[float] = []
    names: list[str] = []

    for row in rows:
        x = row.get("num_active_electrodes")
        y = row.get("loss")
        if x is None or y is None:
            continue
        x_vals.append(float(x))
        y_vals.append(float(y))
        names.append(str(row.get("sample_index", "?")))

    if len(x_vals) < 2:
        return

    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    ax.scatter(x, y, s=20, alpha=0.75)

    for name, xv, yv in zip(names, x, y):
        ax.annotate(name, (xv, yv), textcoords="offset points", xytext=(4, 3), fontsize=7, alpha=0.85)

    if np.std(x) > 1e-12:
        coef = np.polyfit(x, y, deg=1)
        xline = np.linspace(float(x.min()), float(x.max()), 200)
        yline = coef[0] * xline + coef[1]
        ax.plot(xline, yline, linewidth=1.3)

    ax.set_xlabel("Active Electrode Count")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Active Electrode Count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _load_data_splits() -> tuple[Subset, Subset, Subset]:
    train_set, val_set, test_set = load_letters_phosphene_splits(
        npz_path=MAPS_NPZ,
        seed=SEED,
        val_ratio_from_train=VAL_RATIO_FROM_TRAIN,
        max_train_samples=MAX_TRAIN_SAMPLES,
        max_val_samples=MAX_VAL_SAMPLES,
        max_test_samples=MAX_TEST_SAMPLES,
    )
    return train_set, val_set, test_set


def _build_loaders(train_set: Subset, val_set: Subset, test_set: Subset, device: torch.device) -> tuple[DataLoader, DataLoader, DataLoader]:
    use_cuda = device.type == "cuda"
    num_workers = min(4, os.cpu_count() or 1) if use_cuda else _get_cpu_worker_count()

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader


def _train_single(subject_id: str, hemisphere: str, train_dir: Path, device: torch.device) -> Path:
    train_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = train_dir / "checkpoints"
    model_path = train_dir / "single_subject_inverse_model.pt"
    report_path = train_dir / "report.json"

    if SKIP_IF_EXISTS and model_path.exists() and report_path.exists():
        print(f"[SKIP][train] {subject_id}_{hemisphere}: existing outputs found")
        return model_path

    train_set, val_set, test_set = _load_data_splits()
    train_loader, val_loader, test_loader = _build_loaders(train_set, val_set, test_set, device)

    model = InverseModel(in_channels=1, latent_dim=256, electrode_dim=1000, map_size=MAP_SIZE)
    simulator = DifferentiableSimulatorIndependent(
        data_dir=PROJECT_ROOT / "data" / "fmri" / subject_id,
        hemisphere=hemisphere,
        map_size=MAP_SIZE,
    )

    train_config = TrainConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        shared_params_lr=SHARED_PARAMS_LR,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        allow_nondiff_training=False,
        refinement_steps=0,
        refinement_lr=1e-2,
        scheduler_patience=SCHEDULER_PATIENCE,
        scheduler_factor=SCHEDULER_FACTOR,
        early_stop_min_delta=EARLY_STOP_MIN_DELTA,
        monitor_metric=MONITOR_METRIC,
        monitor_mode=MONITOR_MODE,
        min_epochs_for_early_stop=MIN_EPOCHS,
        patience_for_early_stop=EARLY_STOP_PATIENCE,
    )
    loss_config = LossConfig()

    history = train_inverse_model(
        model=model,
        simulator=simulator,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_config=loss_config,
        train_config=train_config,
        valid_electrode_mask=None,
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint=None,
    )

    best_ckpt_path = checkpoint_dir / "checkpoint_best.pt"
    if best_ckpt_path.exists():
        _ = load_checkpoint(best_ckpt_path, model, device)

    test_metrics = evaluate_inverse_model(
        model=model.to(device),
        simulator=simulator.to(device),
        data_loader=test_loader,
        loss_config=loss_config,
        device=device,
        valid_electrode_mask=None,
    )
    random_baseline = evaluate_random_baseline(simulator=simulator.to(device), data_loader=test_loader)
    four_param_baseline = evaluate_four_param_baseline(
        model=model.to(device), simulator=simulator.to(device), data_loader=test_loader
    )

    torch.save(model.state_dict(), model_path)
    sp = model.shared_params.detach().cpu().tolist()
    learned_params = {
        "alpha": sp[0],
        "beta": sp[1],
        "offset_from_base": sp[2],
        "shank_length": sp[3],
    }
    report = {
        "subject_id": subject_id,
        "hemisphere": hemisphere,
        "learned_implant_params": learned_params,
        "history": history,
        "test_metrics": test_metrics,
        "baselines": {
            "random": random_baseline,
            "four_params_only": four_param_baseline,
        },
        "config": {
            "train": train_config.__dict__,
            "loss": loss_config.__dict__,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE][train] {subject_id}_{hemisphere} -> {train_dir}")
    return model_path


def _infer_single(subject_id: str, hemisphere: str, model_path: Path, infer_dir: Path, device: torch.device) -> Path:
    infer_dir.mkdir(parents=True, exist_ok=True)
    summary_path = infer_dir / "summary.json"
    per_sample_jsonl_path = infer_dir / "per_sample_metrics.jsonl"

    if SKIP_IF_EXISTS and summary_path.exists() and per_sample_jsonl_path.exists():
        print(f"[SKIP][infer] {subject_id}_{hemisphere}: existing outputs found")
        return infer_dir

    test_ds = PhospheneDataset.from_phosphene_npz(npz_path=MAPS_NPZ, key="test_phosphenes")
    n_samples = len(test_ds)
    if n_samples == 0:
        raise RuntimeError("No test samples found for inference")

    model = InverseModel(in_channels=1, latent_dim=256, electrode_dim=1000, map_size=MAP_SIZE)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    simulator = DifferentiableSimulatorIndependent(
        data_dir=PROJECT_ROOT / "data" / "fmri" / subject_id,
        hemisphere=hemisphere,
        map_size=MAP_SIZE,
    ).to(device).eval()

    rows: list[dict[str, Any]] = []
    all_dc: list[float] = []
    all_y: list[float] = []
    all_hd: list[float] = []
    all_score: list[float] = []
    all_loss: list[float] = []
    all_num_active: list[float] = []
    all_active_ratio: list[float] = []

    with torch.no_grad():
        for idx in range(n_samples):
            target = test_ds[idx].unsqueeze(0).to(device)

            params, electrode_logits = model(target)
            recon = simulator(params, electrode_logits)

            dc_metric = float(dice_score(recon, target).item())
            hd_metric = float(hellinger_distance(recon, target).item())
            y_metric = float(y_metric_from_params(simulator, params).item())
            score = dc_metric + 0.1 * y_metric - hd_metric
            loss = 2.0 - score

            params_vec = params[0].detach().cpu().numpy().tolist()
            elec_prob = torch.sigmoid(electrode_logits)[0].detach().cpu().numpy()
            num_active = int((elec_prob > 0.5).sum())
            active_ratio = float(num_active / elec_prob.size)

            row = {
                "sample_index": idx,
                "learned_implant_params": {
                    "alpha": params_vec[0],
                    "beta": params_vec[1],
                    "offset_from_base": params_vec[2],
                    "shank_length": params_vec[3],
                },
                "electrode_stats": {
                    "on_threshold": 0.5,
                    "num_active_electrodes": num_active,
                    "active_ratio": active_ratio,
                    "mean_activation": float(elec_prob.mean()),
                },
                "performance": {
                    "dc_metric": dc_metric,
                    "y_metric": y_metric,
                    "hd_metric": hd_metric,
                    "score": score,
                    "loss": loss,
                },
            }
            rows.append(row)
            all_dc.append(dc_metric)
            all_y.append(y_metric)
            all_hd.append(hd_metric)
            all_score.append(score)
            all_loss.append(loss)
            all_num_active.append(float(num_active))
            all_active_ratio.append(active_ratio)

    summary = {
        "subject_id": subject_id,
        "hemisphere": hemisphere,
        "model_path": str(model_path),
        "retinotopy_dir": str(PROJECT_ROOT / "data" / "fmri" / subject_id),
        "map_size": MAP_SIZE,
        "target_source": str(MAPS_NPZ),
        "emnist_split": "test",
        "num_samples": n_samples,
        "aggregate_performance": {
            "dc_metric": _stat(all_dc),
            "y_metric": _stat(all_y),
            "hd_metric": _stat(all_hd),
            "score": _stat(all_score),
            "loss": _stat(all_loss),
        },
        "aggregate_electrode_stats": {
            "on_threshold": 0.5,
            "num_active_electrodes": _stat(all_num_active),
            "active_ratio": _stat(all_active_ratio),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with per_sample_jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[DONE][infer] {subject_id}_{hemisphere} -> {infer_dir}")
    return infer_dir


def _analysis_single(subject_id: str, hemisphere: str, infer_dir: Path, analysis_dir: Path) -> Path:
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_txt = analysis_dir / "overall_summary.txt"
    hist_png = analysis_dir / "hist_all_metrics_grid.png"
    scatter_png = analysis_dir / "scatter_loss_vs_active_electrodes.png"
    table_csv = analysis_dir / "aggregate_table.csv"

    if SKIP_IF_EXISTS and summary_txt.exists() and hist_png.exists() and scatter_png.exists() and table_csv.exists():
        print(f"[SKIP][analysis] {subject_id}_{hemisphere}: existing outputs found")
        return analysis_dir

    per_sample_jsonl = infer_dir / "per_sample_metrics.jsonl"
    if not per_sample_jsonl.exists():
        raise FileNotFoundError(f"Missing inference JSONL: {per_sample_jsonl}")

    rows: list[dict[str, Any]] = []
    with per_sample_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            perf = item.get("performance", {})
            elec = item.get("electrode_stats", {})
            params = item.get("learned_implant_params", {})
            rows.append(
                {
                    "sample_index": item.get("sample_index"),
                    "dc_metric": float(perf.get("dc_metric", 0.0)),
                    "y_metric": float(perf.get("y_metric", 0.0)),
                    "hd_metric": float(perf.get("hd_metric", 0.0)),
                    "score": float(perf.get("score", 0.0)),
                    "loss": float(perf.get("loss", 0.0)),
                    "num_active_electrodes": float(elec.get("num_active_electrodes", 0.0)),
                    "active_ratio": float(elec.get("active_ratio", 0.0)),
                    "mean_activation": float(elec.get("mean_activation", 0.0)),
                    "alpha": float(params.get("alpha", 0.0)),
                    "beta": float(params.get("beta", 0.0)),
                    "offset_from_base": float(params.get("offset_from_base", 0.0)),
                    "shank_length": float(params.get("shank_length", 0.0)),
                }
            )

    columns: dict[str, np.ndarray] = {}
    if rows:
        for key in rows[0].keys():
            vals = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
            if vals:
                columns[key] = np.asarray(vals, dtype=np.float64)

    lines: list[str] = []
    lines.append(f"PhosOpt Analysis: {subject_id}_{hemisphere}")
    lines.append("=" * 80)
    lines.append(f"Input dir: {infer_dir}")
    lines.append(f"Num rows : {len(rows)}")
    lines.append("")

    for key in [
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
    ]:
        if key not in columns:
            continue
        arr = columns[key]
        lines.append(f"[{key}]")
        lines.append(f"  count  : {arr.size}")
        lines.append(f"  mean   : {arr.mean():.6f}")
        lines.append(f"  std    : {arr.std():.6f}")
        lines.append(f"  min    : {arr.min():.6f}")
        lines.append(f"  max    : {arr.max():.6f}")
        lines.append("")

    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    _save_hist_grid(columns, hist_png)
    _save_scatter(rows, scatter_png)

    with table_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "min", "max"])
        for key in [
            "dc_metric",
            "hd_metric",
            "y_metric",
            "score",
            "loss",
            "num_active_electrodes",
            "active_ratio",
            "mean_activation",
        ]:
            if key not in columns:
                continue
            arr = columns[key]
            writer.writerow([key, float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())])

    print(f"[DONE][analysis] {subject_id}_{hemisphere} -> {analysis_dir}")
    return analysis_dir


def main() -> None:
    start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = _get_device()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output root: {OUTPUT_ROOT}")

    runs = [(s, h) for s in SUBJECT_IDS for h in HEMISPHERES]
    print(f"[INFO] Total runs: {len(runs)}")

    for i, (subject_id, hemisphere) in enumerate(runs, start=1):
        run_name = f"{subject_id}_{hemisphere}"
        run_root = OUTPUT_ROOT / run_name
        train_dir = run_root / "train"
        infer_dir = run_root / "infer"
        analysis_dir = run_root / "analysis"

        print("\n" + "=" * 80)
        print(f"[RUN {i}/{len(runs)}] {run_name}")
        print("=" * 80)

        model_path = _train_single(subject_id=subject_id, hemisphere=hemisphere, train_dir=train_dir, device=device)
        _ = _infer_single(subject_id=subject_id, hemisphere=hemisphere, model_path=model_path, infer_dir=infer_dir, device=device)
        _ = _analysis_single(subject_id=subject_id, hemisphere=hemisphere, infer_dir=infer_dir, analysis_dir=analysis_dir)

    elapsed = time.time() - start
    print("\n" + "=" * 80)
    print("[ALL DONE]")
    print(f"Elapsed: {elapsed:.2f} sec")
    print(f"Saved under: {OUTPUT_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
