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

# Unbuffer stdout so prints appear immediately
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

print("[PhosOpt] Loading dependencies...", flush=True)
import argparse
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
from simulator.physics_forward_torch import DifferentiableSimulator
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train inverse phosphene estimator")
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=None,
        help="Directory containing target map .npy files (overrides --maps-npz if set)",
    )
    parser.add_argument(
        "--maps-npz",
        type=Path,
        default=PROJECT_ROOT / "data" / "letters" / "emnist_letters_phosphenes.npz",
        help="Path to .npz containing phosphene maps (default: data/letters/emnist_letters_phosphenes.npz)",
    )
    parser.add_argument(
        "--maps-npz-train-key",
        type=str,
        default="train_phosphenes",
        help="Train key for maps npz (used only for generic npz path).",
    )
    parser.add_argument(
        "--maps-npz-test-key",
        type=str,
        default="test_phosphenes",
        help="Test key for maps npz (used only for generic npz path).",
    )
    parser.add_argument(
        "--val-ratio-from-train",
        type=float,
        default=0.1,
        help="Validation split ratio from train split when using maps-npz.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for train samples")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional cap for val samples")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional cap for test samples")
    parser.add_argument("--subject-id", type=str, default="single_subject", help="Single-subject run identifier")
    parser.add_argument(
        "--retinotopy-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "fmri" / "100610",
        help="Retinotopy directory with mgz files (default: data/fmri/100610)",
    )
    parser.add_argument("--hemisphere", choices=["LH", "RH"], default="LH")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--shared-params-lr", type=float, default=1e-2, help="Learning rate for shared implant params")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "data" / "output" / "inverse_training_filtered")
    parser.add_argument("--valid-electrode-mask", type=Path, default=None, help="Optional 1000-dim mask .npy")
    parser.add_argument(
        "--simulator",
        choices=["diff", "numpy"],
        default="diff",
        help="Simulator backend: 'diff' (differentiable torch, default) or 'numpy' (debug only).",
    )
    parser.add_argument("--map-size", type=int, default=256, help="Output map resolution for diff simulator")
    parser.add_argument(
        "--allow-nondiff-training",
        action="store_true",
        help="Allow training with numpy simulator (debug only).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint .pt file. Default: auto-detect from save-dir.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force fresh training, ignore existing checkpoints.",
    )
    return parser.parse_args()


def _load_valid_electrode_mask(path: Path | None, device: torch.device) -> torch.Tensor | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"valid electrode mask file not found: {path}")
    mask = torch.from_numpy(np.load(path).astype("float32").reshape(-1)).to(device)
    if mask.numel() != 1000:
        raise ValueError(f"valid electrode mask must be 1000-dim, got {mask.numel()}")
    return mask.unsqueeze(0)


def _print_runtime_info(device: torch.device, num_workers: int) -> None:
    print("=== Runtime Info ===")
    print(f"Torch version: {torch.__version__}")
    print(f"Selected device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"DataLoader workers: {num_workers}")
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        print(f"CUDA device index: {idx}")
        print(f"CUDA device name: {torch.cuda.get_device_name(idx)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(idx) / (1024**2):.2f} MiB")
    print("====================")


def main() -> None:
    print("Starting PhosOpt training...", flush=True)
    args = parse_args()
    torch.manual_seed(args.seed)
    device = _get_device()
    use_cuda = device.type == "cuda"
    print(f"Device: {device}", flush=True)

    if not use_cuda:
        torch.set_num_threads(_get_cpu_worker_count())

    if args.maps_dir is not None:
        print(f"Loading dataset from directory: {args.maps_dir}", flush=True)
        dataset = PhospheneDataset.from_npy_dir(args.maps_dir)
        train_set, val_set, test_set = make_splits(
            dataset,
            SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=args.seed),
        )
    elif args.maps_npz is not None:
        print(f"Loading dataset from npz: {args.maps_npz}", flush=True)
        # Preferred path for letters phosphene dataset.
        if args.maps_npz_test_key == "test_phosphenes" and args.maps_npz_train_key == "train_phosphenes":
            train_set, val_set, test_set = load_letters_phosphene_splits(
                npz_path=args.maps_npz,
                seed=args.seed,
                val_ratio_from_train=args.val_ratio_from_train,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
                max_test_samples=args.max_test_samples,
            )
        else:
            # Generic npz loader path with user-defined keys.
            train_ds = PhospheneDataset.from_phosphene_npz(args.maps_npz, key=args.maps_npz_train_key)
            test_ds = PhospheneDataset.from_phosphene_npz(args.maps_npz, key=args.maps_npz_test_key)
            idx = np.arange(len(train_ds))
            rng = np.random.default_rng(args.seed)
            rng.shuffle(idx)
            n_val = int(round(len(idx) * args.val_ratio_from_train))
            n_val = max(1, min(n_val, len(idx) - 1))
            val_idx = idx[:n_val].tolist()
            train_idx = idx[n_val:].tolist()
            if args.max_train_samples is not None:
                train_idx = train_idx[: max(1, int(args.max_train_samples))]
            train_set = Subset(train_ds, train_idx)
            val_set = Subset(train_ds, val_idx)
            if args.max_test_samples is not None:
                test_idx = list(range(min(len(test_ds), max(1, int(args.max_test_samples)))))
                test_set = Subset(test_ds, test_idx)
            else:
                test_set = test_ds
    else:
        raise ValueError("Either --maps-dir or --maps-npz must be provided")
    num_workers = 0 if use_cuda else _get_cpu_worker_count()
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    print(
        f"Dataset split sizes -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}",
        flush=True,
    )
    _print_runtime_info(device=device, num_workers=num_workers)

    print("Building model...", flush=True)
    model = InverseModel(in_channels=1, latent_dim=256, electrode_dim=1000)

    if args.simulator == "diff":
        print(f"Loading retinotopy data from {args.retinotopy_dir}...", flush=True)
        print(f"Using differentiable simulator (map_size={args.map_size})", flush=True)
        simulator = DifferentiableSimulator(
            data_dir=args.retinotopy_dir,
            hemisphere=args.hemisphere,
            map_size=args.map_size,
        )
    else:
        print("Using numpy simulator (non-differentiable, debug only)", flush=True)
        simulator = NumpySimulatorAdapter(data_dir=args.retinotopy_dir, hemisphere=args.hemisphere)
    loss_config = LossConfig(
        recon_weight=1.0,
        dice_weight=0.5,
        sparsity_weight=1e-3,
        param_prior_weight=1e-4,
        invalid_region_weight=1e-3,
        warmup_epochs=10,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        shared_params_lr=args.shared_params_lr,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        allow_nondiff_training=args.allow_nondiff_training,
        refinement_steps=0,
        refinement_lr=1e-2,
    )

    valid_mask = _load_valid_electrode_mask(args.valid_electrode_mask, device=device)

    # Checkpoint directory (always active)
    checkpoint_dir = args.save_dir / f"{args.subject_id}_checkpoints"

    # Auto-resume: find existing checkpoint unless --no-resume
    resume_ckpt = None
    if not args.no_resume:
        ckpt_path = args.resume
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
            print(f"Checkpoint loaded: epoch {completed}/{args.epochs} completed", flush=True)
            if completed >= args.epochs:
                print(f"Already completed {completed} epochs (target={args.epochs}). "
                      f"Use --epochs {completed + 10} to train more, or --no-resume for fresh start.")
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

    test_metrics = evaluate_inverse_model(
        model=model.to(device), simulator=simulator.to(device), data_loader=test_loader, device=device
    )
    random_baseline = evaluate_random_baseline(simulator=simulator.to(device), data_loader=test_loader)
    four_param_baseline = evaluate_four_param_baseline(
        model=model.to(device), simulator=simulator.to(device), data_loader=test_loader
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.save_dir / f"{args.subject_id}_inverse_model.pt"
    report_path = args.save_dir / f"{args.subject_id}_report.json"
    torch.save(model.state_dict(), model_path)
    sp = model.shared_params.detach().cpu().tolist()
    learned_params = {
        "alpha": sp[0], "beta": sp[1],
        "offset_from_base": sp[2], "shank_length": sp[3],
    }
    report = {
        "subject_id": args.subject_id,
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
