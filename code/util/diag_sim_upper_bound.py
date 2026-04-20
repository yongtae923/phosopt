"""
Simulator upper-bound test: directly optimize 4 params + 1000 electrode logits
to reconstruct a single target map, bypassing the model entirely.

If direct optimization can't reduce MSE significantly, then the simulator
cannot express the targets, and no model architecture will help.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import normalize_target_map
from simulator.physics_forward_torch import DifferentiableSimulator


def _get_map_size_from_env(default: int = 256) -> int:
    raw = os.getenv("PHOSOPT_MAP_SIZE", str(default)).strip()
    try:
        size = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid PHOSOPT_MAP_SIZE='{raw}'. Expected integer.") from exc
    if size < 8 or size % 8 != 0:
        raise ValueError(f"PHOSOPT_MAP_SIZE must be divisible by 8 and >= 8, got {size}")
    return size


def _default_letters_npz_for_map_size(project_root: Path, map_size: int) -> Path:
    letters_dir = project_root / "data" / "letters"
    if map_size == 256:
        return letters_dir / "emnist_letters_v3_halfright.npz"
    return letters_dir / f"emnist_letters_v3_halfright_{map_size}.npz"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    map_size = _get_map_size_from_env(default=256)
    npz_path = Path(
        os.getenv(
            "PHOSOPT_MAPS_NPZ",
            str(_default_letters_npz_for_map_size(project_root, map_size)),
        )
    )
    retino_dir = project_root / "data" / "fmri" / "100610"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_targets = 4
    n_steps = 500
    lr = 0.05

    # Load targets
    with np.load(npz_path) as f:
        raw = f["train_phosphenes"][:n_targets].astype(np.float32)
    if raw.ndim == 4 and raw.shape[1] == 1:
        raw = raw[:, 0, :, :]
    normed = np.stack([normalize_target_map(m) for m in raw])
    target = torch.from_numpy(normed).unsqueeze(1).to(device)
    print(f"Target shape: {target.shape}, range: [{target.min():.3f}, {target.max():.3f}]")

    # Init simulator
    sim = DifferentiableSimulator(data_dir=retino_dir, hemisphere="LH", map_size=map_size)
    sim = sim.to(device).eval()

    # Directly optimize parameters (no model)
    params = torch.randn(n_targets, 4, device=device, requires_grad=True)
    electrode_logits = torch.zeros(n_targets, 1000, device=device, requires_grad=True)
    opt = Adam([params, electrode_logits], lr=lr)

    print(f"\nDirect optimization: {n_targets} targets, {n_steps} steps, lr={lr}")
    print("-" * 70)

    for step in range(n_steps):
        recon = sim(params, electrode_logits)
        mse = torch.mean((recon - target) ** 2)
        per_sample_mse = ((recon - target) ** 2).flatten(start_dim=1).mean(dim=1)

        opt.zero_grad()
        mse.backward()
        opt.step()

        if step % 50 == 0 or step == n_steps - 1:
            elec_prob = torch.sigmoid(electrode_logits)
            bounded_params = params.detach()
            print(
                f"  step {step:4d} | MSE={mse.item():.6f} | "
                f"per-sample: {[f'{v:.4f}' for v in per_sample_mse.tolist()]} | "
                f"elec_prob: [{elec_prob.min():.3f}, {elec_prob.max():.3f}] mean={elec_prob.mean():.3f} | "
                f"recon: [{recon.min():.3f}, {recon.max():.3f}]"
            )

    # Final comparison
    print("\n" + "=" * 70)
    print("UPPER BOUND RESULT")
    print("=" * 70)
    final_recon = sim(params, electrode_logits).detach()
    final_mse = torch.mean((final_recon - target) ** 2).item()
    zero_mse = torch.mean(target ** 2).item()
    print(f"  Final MSE (direct opt):  {final_mse:.6f}")
    print(f"  MSE of predicting zeros: {zero_mse:.6f}")
    print(f"  Improvement over zeros:  {(1 - final_mse / zero_mse) * 100:.1f}%")
    print(f"  Recon range: [{final_recon.min():.4f}, {final_recon.max():.4f}]")
    print(f"  Recon mean:  {final_recon.mean():.6f}")
    print(f"  Target mean: {target.mean():.6f}")

    if final_mse > 0.10:
        print("\n  >>> DIAGNOSIS: Simulator CANNOT express target patterns.")
        print("  >>> The ~0.127 MSE floor in training is a SIMULATOR limitation,")
        print("  >>> not a model architecture issue.")
    elif final_mse < 0.01:
        print("\n  >>> DIAGNOSIS: Simulator CAN express targets well.")
        print("  >>> The training issue is in the model/optimizer, not the simulator.")
    else:
        print(f"\n  >>> DIAGNOSIS: Simulator can partially express targets (MSE={final_mse:.4f}).")
        print("  >>> Some improvement possible, but simulator is a limiting factor.")


if __name__ == "__main__":
    main()
