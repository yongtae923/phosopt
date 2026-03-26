# D:\yongtae\phosopt\code\infer.py

"""
Inference script for PhosOpt inverse model.

Given:
- a trained inverse model checkpoint (.pt),
- a target phosphene map (single .npy) or an index from the EMNIST .npz,
- and the retinotopy directory for the simulator,

this script:
1) Loads the trained InverseModel and DifferentiableSimulator.
2) Runs a forward pass to estimate:
   - 4 implant parameters (alpha, beta, offset_from_base, shank_length),
   - 1000 electrode activations.
3) Reconstructs the phosphene map via the simulator.
4) Saves all results to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset import PhospheneDataset  # noqa: E402
from models.inverse_model import InverseModel  # noqa: E402
from simulator.physics_forward_torch import DifferentiableSimulator  # noqa: E402


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda")
            _ = t + t
            return torch.device("cuda")
        except Exception:
            print("[WARNING] CUDA detected but GPU kernels failed. Falling back to CPU.")
    return torch.device("cpu")


def _load_single_from_npy(path: Path) -> np.ndarray:
    arr = np.load(path).astype("float32")
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected [H,W] or [1,H,W] for npy target, got shape {arr.shape}")
    return arr


def _load_single_from_emnist_npz(npz_path: Path, split: str, index: int) -> np.ndarray:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    key = "train_phosphenes" if split == "train" else "test_phosphenes"
    ds = PhospheneDataset.from_phosphene_npz(npz_path=npz_path, key=key)
    if index < 0 or index >= len(ds):
        raise IndexError(f"index {index} out of range for split '{split}' (len={len(ds)})")
    x = ds[index]  # tensor [1,H,W], already normalized in dataset
    return x.squeeze(0).numpy()


def _normalize_map(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = m.astype("float32")
    max_val = float(m.max())
    if max_val > eps:
        m = m / max_val
    return m


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PhosOpt inverse inference script")

    # Model / simulator
    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained inverse model .pt (e.g., data/output/inverse_training/sXXXX_inverse_model.pt)",
    )
    p.add_argument(
        "--retinotopy-dir",
        type=Path,
        required=True,
        help="Retinotopy directory used during training (e.g., data/fmri/100610)",
    )
    p.add_argument(
        "--hemisphere",
        choices=["LH", "RH"],
        default="LH",
        help="Hemisphere used during training (default: LH)",
    )
    p.add_argument(
        "--map-size",
        type=int,
        default=256,
        help="Output map resolution for DifferentiableSimulator (must match training, default: 256)",
    )

    # Target phosphene source (one of: --target-npy or --emnist-npz + --emnist-split + --emnist-index)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--target-npy",
        type=Path,
        help="Path to single phosphene map .npy file ([H,W] or [1,H,W])",
    )
    group.add_argument(
        "--emnist-npz",
        type=Path,
        help="Path to EMNIST phosphene .npz (train_phosphenes / test_phosphenes)",
    )
    p.add_argument(
        "--emnist-split",
        choices=["train", "test"],
        default="test",
        help="Which EMNIST split to draw from when using --emnist-npz (default: test)",
    )
    p.add_argument(
        "--emnist-index",
        type=int,
        default=0,
        help="Index into EMNIST split when using --emnist-npz (default: 0)",
    )

    # Output
    p.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Directory to save outputs (JSON + NPY)",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="sample",
        help="Tag/prefix for saved files (default: sample)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = _get_device()
    print(f"[PhosOpt][infer] Using device: {device}")

    # ------------------------------------------------------------------
    # Load target phosphene map
    # ------------------------------------------------------------------
    if args.target_npy is not None:
        print(f"[PhosOpt][infer] Loading target from npy: {args.target_npy}")
        target_np = _load_single_from_npy(args.target_npy)
    else:
        if args.emnist_npz is None:
            raise ValueError("Either --target-npy or --emnist-npz must be provided.")
        print(
            f"[PhosOpt][infer] Loading target from EMNIST npz: {args.emnist_npz}, "
            f"split={args.emnist_split}, index={args.emnist_index}"
        )
        target_np = _load_single_from_emnist_npz(
            npz_path=args.emnist_npz,
            split=args.emnist_split,
            index=args.emnist_index,
        )

    target_np = _normalize_map(target_np)
    H, W = target_np.shape
    print(f"[PhosOpt][infer] Target map shape (H,W): {H}x{W}")
    target = torch.from_numpy(target_np).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

    # ------------------------------------------------------------------
    # Load model and simulator
    # ------------------------------------------------------------------
    print(f"[PhosOpt][infer] Loading model from: {args.model_path}")
    model = InverseModel(in_channels=1, latent_dim=256, electrode_dim=1000)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    print(f"[PhosOpt][infer] Initializing DifferentiableSimulator from {args.retinotopy_dir}")
    simulator = DifferentiableSimulator(
        data_dir=args.retinotopy_dir,
        hemisphere=args.hemisphere,
        map_size=args.map_size,
    ).to(device).eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    with torch.no_grad():
        params, electrode_logits = model(target)
        recon = simulator(params, electrode_logits)

    params_vec = params[0].detach().cpu().numpy().tolist()
    elec_prob = torch.sigmoid(electrode_logits)[0].detach().cpu().numpy()
    recon_np = recon[0, 0].detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    args.save_dir.mkdir(parents=True, exist_ok=True)
    base = args.save_dir / args.tag

    params_path = base.with_suffix(".params.json")
    elec_path = base.with_suffix(".electrodes.npy")
    recon_path = base.with_suffix(".recon.npy")
    target_path = base.with_suffix(".target.npy")

    learned_params = {
        "alpha": params_vec[0],
        "beta": params_vec[1],
        "offset_from_base": params_vec[2],
        "shank_length": params_vec[3],
    }
    meta = {
        "model_path": str(args.model_path),
        "retinotopy_dir": str(args.retinotopy_dir),
        "hemisphere": args.hemisphere,
        "map_size": args.map_size,
        "target_source": str(args.target_npy or args.emnist_npz),
        "emnist_split": args.emnist_split if args.emnist_npz is not None else None,
        "emnist_index": args.emnist_index if args.emnist_npz is not None else None,
        "learned_implant_params": learned_params,
    }

    params_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(elec_path, elec_prob)
    np.save(recon_path, recon_np)
    np.save(target_path, target_np)

    print(f"[PhosOpt][infer] Saved implant params JSON: {params_path}")
    print(f"[PhosOpt][infer] Saved electrode probabilities NPY: {elec_path}")
    print(f"[PhosOpt][infer] Saved reconstructed map NPY: {recon_path}")
    print(f"[PhosOpt][infer] Saved (normalized) target map NPY: {target_path}")


if __name__ == "__main__":
    main()

