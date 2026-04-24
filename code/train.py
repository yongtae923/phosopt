# D:\yongtae\phosopt\code\train.py

"""
Training script for PhosOpt inverse model.

Given:
- a dataset of target phosphene maps (directory of .npy or .npz with
    train/test splits),
- the retinotopy directory for the simulator,

this script:
1) Loads the dataset and creates train/val/test splits.
2) Builds the InverseModel and DifferentiableSimulator.
3) Trains the model to estimate implant parameters and electrode activations.
4) Evaluates the trained model on the test set and saves results to disk.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Unbuffer stdout so prints appear immediately
_stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(_stdout_reconfigure):
    _stdout_reconfigure(line_buffering=True)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

print("[PhosOpt] Loading dependencies...", flush=True)
import json
import sys
from pathlib import Path

import numpy as np
print("[PhosOpt] Loading torch...", flush=True)
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

print("[PhosOpt] Loading project modules...", flush=True)
from dataset import PhospheneDataset, SplitConfig, load_letters_phosphene_splits, make_splits
from loss.losses import LossConfig
from models.inverse_model import InverseModel
from simulator.physics_forward_torch_v2 import DifferentiableSimulatorIndependent
from simulator.simulator_wrapper import NumpySimulatorAdapter
print("[PhosOpt] Imports done.", flush=True)
from trainer import (
    TrainConfig,
    evaluate_four_param_baseline,
    evaluate_inverse_model,
    evaluate_random_baseline,
    load_checkpoint,
    train_inverse_model,
)


# -----------------------------------------------------------------------------
# Training configuration (edit here instead of CLI args)
# -----------------------------------------------------------------------------
MAPS_DIR = None
MAPS_NPZ = PROJECT_ROOT / "data" / "letters" / "emnist_letters_v3_halfright_128.npz"
MAPS_NPZ_TRAIN_KEY = "train_phosphenes"
MAPS_NPZ_TEST_KEY = "test_phosphenes"
VAL_RATIO_FROM_TRAIN = 0.1
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None
SUBJECT_ID = "single_subject"
RETINOTOPY_DIR = PROJECT_ROOT / "data" / "fmri" / "100610"
HEMISPHERE = "LH"
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
SAVE_DIR = PROJECT_ROOT / "data" / "output" / "0422_model"
VALID_ELECTRODE_MASK = None
SIMULATOR = "diff"
MAP_SIZE = 128
ALLOW_NONDIFF_TRAINING = False
RESUME = None
NO_RESUME = False
DEFAULT_GPU_MEM_FRACTION = 1


def _configure_cuda_memory_fraction() -> float | None:
    """Apply per-process CUDA memory cap.

    Uses GPU_MEM_FRACTION env var when set, otherwise defaults to 0.90.
    """
    raw = os.getenv("GPU_MEM_FRACTION")
    if raw is None or raw.strip() == "":
        fraction = DEFAULT_GPU_MEM_FRACTION
    else:
        try:
            fraction = float(raw)
        except ValueError:
            print(f"[WARNING] Invalid GPU_MEM_FRACTION='{raw}'. Expected a float in (0, 1].")
            return None

    if not (0.0 < fraction <= 1.0):
        print(f"[WARNING] GPU_MEM_FRACTION out of range: {fraction}. Expected (0, 1].")
        return None

    if not torch.cuda.is_available():
        return None

    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        return fraction
    except Exception as exc:
        print(f"[WARNING] Failed to apply GPU_MEM_FRACTION={fraction}: {exc}")
        return None


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


def _load_valid_electrode_mask(path: Path | None, device: torch.device) -> torch.Tensor | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"valid electrode mask file not found: {path}")
    mask = torch.from_numpy(np.load(path).astype("float32").reshape(-1)).to(device)
    if mask.numel() != 1000:
        raise ValueError(f"valid electrode mask must be 1000-dim, got {mask.numel()}")
    return mask.unsqueeze(0)


def _print_runtime_info(device: torch.device, num_workers: int, gpu_mem_fraction: float | None) -> None:
    print("=== Runtime Info ===")
    print(f"Torch version: {torch.__version__}")
    print(f"Selected device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"DataLoader workers: {num_workers}")
    if gpu_mem_fraction is not None:
        print(f"GPU memory cap (per-process): {gpu_mem_fraction:.2f}")
    else:
        print("GPU memory cap (per-process): not set")
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        print(f"CUDA device index: {idx}")
        print(f"CUDA device name: {torch.cuda.get_device_name(idx)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(idx) / (1024**2):.2f} MiB")
    print("====================")


def main() -> None:
    print("Starting PhosOpt training...", flush=True)
    torch.manual_seed(SEED)
    gpu_mem_fraction = _configure_cuda_memory_fraction()
    device = _get_device()
    use_cuda = device.type == "cuda"
    print(f"Device: {device}", flush=True)

    if not use_cuda:
        torch.set_num_threads(_get_cpu_worker_count())

    if MAPS_DIR is not None:
        print(f"Loading dataset from directory: {MAPS_DIR}", flush=True)
        dataset = PhospheneDataset.from_npy_dir(MAPS_DIR)
        train_set, val_set, test_set = make_splits(
            dataset,
            SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=SEED),
        )
    elif MAPS_NPZ is not None:
        print(f"Loading dataset from npz: {MAPS_NPZ}", flush=True)
        # Preferred path for letters phosphene dataset.
        if MAPS_NPZ_TEST_KEY == "test_phosphenes" and MAPS_NPZ_TRAIN_KEY == "train_phosphenes":
            train_set, val_set, test_set = load_letters_phosphene_splits(
                npz_path=MAPS_NPZ,
                seed=SEED,
                val_ratio_from_train=VAL_RATIO_FROM_TRAIN,
                max_train_samples=MAX_TRAIN_SAMPLES,
                max_val_samples=MAX_VAL_SAMPLES,
                max_test_samples=MAX_TEST_SAMPLES,
            )
        else:
            # Generic npz loader path with user-defined keys.
            train_ds = PhospheneDataset.from_phosphene_npz(MAPS_NPZ, key=MAPS_NPZ_TRAIN_KEY)
            test_ds = PhospheneDataset.from_phosphene_npz(MAPS_NPZ, key=MAPS_NPZ_TEST_KEY)
            idx = np.arange(len(train_ds))
            rng = np.random.default_rng(SEED)
            rng.shuffle(idx)
            n_val = int(round(len(idx) * VAL_RATIO_FROM_TRAIN))
            n_val = max(1, min(n_val, len(idx) - 1))
            val_idx = idx[:n_val].tolist()
            train_idx = idx[n_val:].tolist()
            if MAX_TRAIN_SAMPLES is not None:
                train_idx = train_idx[: max(1, int(MAX_TRAIN_SAMPLES))]
            train_set = Subset(train_ds, train_idx)
            val_set = Subset(train_ds, val_idx)
            if MAX_TEST_SAMPLES is not None:
                test_idx = list(range(min(len(test_ds), max(1, int(MAX_TEST_SAMPLES)))))
                test_set = Subset(test_ds, test_idx)
            else:
                test_set = test_ds
    else:
        raise ValueError("MAPS_DIR or MAPS_NPZ must be provided")

    sample = train_set[0]
    if not isinstance(sample, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor sample, got {type(sample)!r}")
    if sample.ndim != 3 or sample.shape[0] != 1:
        raise ValueError(f"Expected target sample shape [1,H,W], got {tuple(sample.shape)}")
    target_h, target_w = int(sample.shape[-2]), int(sample.shape[-1])
    if target_h != target_w:
        raise ValueError(f"Target maps must be square, got {target_h}x{target_w}")
    if target_h != MAP_SIZE:
        raise ValueError(
            f"Dataset map size ({target_h}x{target_w}) does not match simulator MAP_SIZE={MAP_SIZE}. "
            "Set MAP_SIZE to the dataset resolution or resize the target maps before training."
        )

    if use_cuda:
        num_workers = min(4, os.cpu_count() or 1)
    else:
        num_workers = _get_cpu_worker_count()
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
    print(
        f"Dataset split sizes -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}",
        flush=True,
    )
    _print_runtime_info(device=device, num_workers=num_workers, gpu_mem_fraction=gpu_mem_fraction)

    print("Building model...", flush=True)
    model = InverseModel(in_channels=1, latent_dim=256, electrode_dim=1000, map_size=MAP_SIZE)

    if SIMULATOR == "diff":
        print(f"Loading retinotopy data from {RETINOTOPY_DIR}...", flush=True)
        print(f"Using differentiable simulator v2 (map_size={MAP_SIZE})", flush=True)
        simulator = DifferentiableSimulatorIndependent(
            data_dir=RETINOTOPY_DIR,
            hemisphere=HEMISPHERE,
            map_size=MAP_SIZE,
        )
    else:
        print("Using numpy simulator (non-differentiable, debug only)", flush=True)
        simulator = NumpySimulatorAdapter(data_dir=RETINOTOPY_DIR, hemisphere=HEMISPHERE)
    loss_config = LossConfig()
    train_config = TrainConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        shared_params_lr=SHARED_PARAMS_LR,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        allow_nondiff_training=ALLOW_NONDIFF_TRAINING,
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

    valid_mask = _load_valid_electrode_mask(VALID_ELECTRODE_MASK, device=device)

    # Checkpoint directory (always active)
    checkpoint_dir = SAVE_DIR / f"{SUBJECT_ID}_checkpoints"

    # Auto-resume: find existing checkpoint unless --no-resume
    resume_ckpt = None
    if not NO_RESUME:
        ckpt_path = RESUME
        if ckpt_path is None:
            auto_path = checkpoint_dir / "checkpoint_latest.pt"
            if auto_path.exists():
                ckpt_path = auto_path
                print(f"[Auto-resume] Found checkpoint: {ckpt_path}", flush=True)
        if ckpt_path is not None:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            print(f"Loading checkpoint: {ckpt_path}", flush=True)
            resume_ckpt = load_checkpoint(ckpt_path, model, device)
            completed = resume_ckpt["epoch"] + 1
            print(f"Checkpoint loaded: epoch {completed}/{EPOCHS} completed", flush=True)
            if completed >= EPOCHS:
                print(f"Already completed {completed} epochs (target={EPOCHS}). "
                      f"Use EPOCHS = {completed + 10} to train more, or set NO_RESUME = True for fresh start.")
                return

    print("Starting training...", flush=True)
    history = train_inverse_model(
        model=model,
        simulator=simulator,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_config=loss_config,
        train_config=train_config,
        valid_electrode_mask=valid_mask,
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint=resume_ckpt,
    )

    best_ckpt_path = checkpoint_dir / "checkpoint_best.pt"
    if best_ckpt_path.exists():
        print(f"Loading best checkpoint for final evaluation: {best_ckpt_path}", flush=True)
        _ = load_checkpoint(best_ckpt_path, model, device)
    else:
        print("[WARNING] Best checkpoint not found. Using last model state for final evaluation.", flush=True)

    test_metrics = evaluate_inverse_model(
        model=model.to(device),
        simulator=simulator.to(device),
        data_loader=test_loader,
        loss_config=loss_config,
        device=device,
        valid_electrode_mask=valid_mask,
    )
    random_baseline = evaluate_random_baseline(simulator=simulator.to(device), data_loader=test_loader)
    four_param_baseline = evaluate_four_param_baseline(
        model=model.to(device), simulator=simulator.to(device), data_loader=test_loader
    )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SAVE_DIR / f"{SUBJECT_ID}_inverse_model.pt"
    report_path = SAVE_DIR / f"{SUBJECT_ID}_report.json"
    torch.save(model.state_dict(), model_path)
    sp = model.shared_params.detach().cpu().tolist()
    learned_params = {
        "alpha": sp[0], "beta": sp[1],
        "offset_from_base": sp[2], "shank_length": sp[3],
    }
    report = {
        "subject_id": SUBJECT_ID,
        "learned_implant_params": learned_params,
        "history": history,
        "test_metrics": test_metrics,
        "baselines": {
            "random": random_baseline,
            "four_params_only": four_param_baseline,
        },
        "config": {
            "loss": loss_config.__dict__,
            "train": train_config.__dict__,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(f"Learned implant params: {learned_params}")
    print(f"Device: {device}")
    print(f"DataLoader workers: {num_workers}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
