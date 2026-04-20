"""
Diagnostic script: traces data through the entire pipeline and prints
statistics at every stage to identify scale/normalization problems.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


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

    # ── 1. Raw data statistics ──
    print("=" * 60)
    print("STAGE 1: Raw data from npz")
    print("=" * 60)
    with np.load(npz_path) as f:
        raw = f["train_phosphenes"][:10].astype(np.float32)
    print(f"  shape: {raw.shape}")
    print(f"  min={raw.min():.6f}  max={raw.max():.6f}  mean={raw.mean():.6f}")
    print(f"  per-sample sums (first 5): {[float(raw[i].sum()) for i in range(5)]}")
    print(f"  non-zero ratio: {(raw != 0).mean():.4f}")

    # ── 2. After normalize_target_map (current: max + sum-to-one) ──
    from dataset import normalize_target_map
    normed = np.stack([normalize_target_map(m) for m in raw])
    print("\n" + "=" * 60)
    print("STAGE 2: After normalize_target_map (max + sum-to-one)")
    print("=" * 60)
    print(f"  min={normed.min():.2e}  max={normed.max():.2e}  mean={normed.mean():.2e}")
    print(f"  per-sample sums (first 5): {[float(normed[i].sum()) for i in range(5)]}")
    print(f"  per-sample max  (first 5): {[float(normed[i].max()) for i in range(5)]}")

    # ── 3. Model output check ──
    print("\n" + "=" * 60)
    print("STAGE 3: Model output (untrained, random weights)")
    print("=" * 60)
    from models import InverseModel
    model = InverseModel(
        in_channels=1,
        latent_dim=128,
        electrode_dim=1000,
        input_map_size=map_size,
    )
    model.eval()

    # Raw data has shape [N, 1, 256, 256]; squeeze channel for normalize, then add back
    raw_squeezed = raw[:, 0, :, :] if raw.ndim == 4 else raw
    normed = np.stack([normalize_target_map(m) for m in raw_squeezed])
    sample = torch.from_numpy(normed[:4]).unsqueeze(1)
    print(f"  Input tensor shape: {sample.shape}")
    print(f"  Input range: [{sample.min():.4f}, {sample.max():.4f}]")
    with torch.no_grad():
        params, electrode_logits = model(sample)
    print(f"  params shape={params.shape}")
    print(f"  params min={params.min():.2f}  max={params.max():.2f}  mean={params.mean():.2f}")
    print(f"  electrode_logits shape={electrode_logits.shape}")
    print(f"  electrode_logits min={electrode_logits.min():.4f}  max={electrode_logits.max():.4f}")
    elec_prob = torch.sigmoid(electrode_logits)
    print(f"  electrode_prob min={elec_prob.min():.4f}  max={elec_prob.max():.4f}  mean={elec_prob.mean():.4f}")

    # ── 4. Simulator output check ──
    print("\n" + "=" * 60)
    print("STAGE 4: Simulator output")
    print("=" * 60)
    from simulator.physics_forward_torch import DifferentiableSimulator
    sim = DifferentiableSimulator(data_dir=retino_dir, hemisphere="LH", map_size=map_size)
    sim.eval()
    with torch.no_grad():
        recon = sim(params, electrode_logits)
    print(f"  recon shape={recon.shape}")
    print(f"  recon min={recon.min():.2e}  max={recon.max():.2e}  mean={recon.mean():.2e}")
    print(f"  recon sum per sample: {[float(recon[i].sum()) for i in range(recon.shape[0])]}")

    # ── 5. Loss diagnostics ──
    print("\n" + "=" * 60)
    print("STAGE 5: Loss values")
    print("=" * 60)
    target_tensor = sample
    mse_val = torch.mean((recon - target_tensor) ** 2)
    print(f"  MSE: {mse_val.item():.4e}")

    # ── 6. Dice threshold analysis ──
    print("\n" + "=" * 60)
    print("STAGE 6: Dice threshold analysis")
    print("=" * 60)
    for thresh in [0.01, 0.001, 1e-4, 1e-5, 1e-6]:
        recon_above = (recon > thresh).float().mean()
        target_above = (target_tensor > thresh).float().mean()
        print(f"  thresh={thresh:.0e}  recon>{thresh}: {recon_above:.4f}  target>{thresh}: {target_above:.4f}")

    # ── 7. Gradient flow check ──
    print("\n" + "=" * 60)
    print("STAGE 7: Gradient flow check")
    print("=" * 60)
    model.train()
    params_g, electrode_logits_g = model(sample)
    recon_g = sim(params_g, electrode_logits_g)
    loss = torch.mean((recon_g - target_tensor) ** 2)
    loss.backward()
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()
    print(f"  Loss value: {loss.item():.4e}")
    print(f"  Non-zero gradients: {sum(1 for v in grad_norms.values() if v > 0)}/{len(grad_norms)}")
    if grad_norms:
        max_grad = max(grad_norms.values())
        min_grad = min(v for v in grad_norms.values() if v > 0) if any(v > 0 for v in grad_norms.values()) else 0
        print(f"  Grad norm range: [{min_grad:.2e}, {max_grad:.2e}]")
        for name, gn in sorted(grad_norms.items(), key=lambda x: -x[1])[:5]:
            print(f"    {name}: {gn:.2e}")

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
