from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset import PhospheneDataset, SplitConfig, load_letters_phosphene_splits, make_splits
from loss.losses import LossConfig
from models.inverse_model import InverseModel
from simulator.simulator_wrapper import NumpySimulatorAdapter
from trainer import (
    TrainConfig,
    evaluate_four_param_baseline,
    evaluate_inverse_model,
    evaluate_random_baseline,
    train_inverse_model,
)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_cpu_worker_count() -> int:
    total = os.cpu_count() or 1
    workers = int(total * 0.8)
    return max(1, workers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train inverse phosphene estimator")
    parser.add_argument("--maps-dir", type=Path, default=None, help="Directory containing target map .npy files")
    parser.add_argument(
        "--maps-npz",
        type=Path,
        default=None,
        help="Optional .npz containing phosphene maps (e.g., emnist_letters_phosphenes.npz)",
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
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional cap for test samples")
    parser.add_argument("--subject-id", type=str, default="single_subject", help="Single-subject run identifier")
    parser.add_argument("--retinotopy-dir", type=Path, required=True, help="Retinotopy directory with mgz files")
    parser.add_argument("--hemisphere", choices=["LH", "RH"], default="LH")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "data" / "output" / "inverse_training")
    parser.add_argument("--valid-electrode-mask", type=Path, default=None, help="Optional 1000-dim mask .npy")
    parser.add_argument(
        "--allow-nondiff-training",
        action="store_true",
        help="Allow training with numpy simulator (debug only).",
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
    args = parse_args()
    torch.manual_seed(args.seed)
    device = _get_device()
    use_cuda = device.type == "cuda"

    if not use_cuda:
        torch.set_num_threads(_get_cpu_worker_count())

    if args.maps_npz is not None:
        # Preferred path for letters phosphene dataset.
        if args.maps_npz_test_key == "test_phosphenes" and args.maps_npz_train_key == "train_phosphenes":
            train_set, val_set, test_set = load_letters_phosphene_splits(
                npz_path=args.maps_npz,
                seed=args.seed,
                val_ratio_from_train=args.val_ratio_from_train,
                max_train_samples=args.max_train_samples,
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
        if args.maps_dir is None:
            raise ValueError("Either --maps-dir or --maps-npz must be provided")
        dataset = PhospheneDataset.from_npy_dir(args.maps_dir)
        train_set, val_set, test_set = make_splits(
            dataset,
            SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=args.seed),
        )
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
        f"Dataset split sizes -> train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}"
    )
    _print_runtime_info(device=device, num_workers=num_workers)

    model = InverseModel(in_channels=1, latent_dim=128, electrode_dim=1000)

    # Default implementation path uses the existing numpy simulator adapter.
    simulator = NumpySimulatorAdapter(data_dir=args.retinotopy_dir, hemisphere=args.hemisphere)
    loss_config = LossConfig(
        recon_weight=1.0,
        sparsity_weight=1e-3,
        param_prior_weight=1e-4,
        invalid_region_weight=1e-3,
        warmup_epochs=10,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip_norm=1.0,
        allow_nondiff_training=args.allow_nondiff_training,
        refinement_steps=0,
        refinement_lr=1e-2,
    )

    valid_mask = _load_valid_electrode_mask(args.valid_electrode_mask, device=device)
    print("Starting training...")
    history = train_inverse_model(
        model=model,
        simulator=simulator,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_config=loss_config,
        train_config=train_config,
        valid_electrode_mask=valid_mask,
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
    report = {
        "subject_id": args.subject_id,
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
    print(f"Device: {device}")
    print(f"DataLoader workers: {num_workers}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
