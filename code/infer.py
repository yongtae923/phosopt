# D:\yongtae\phosopt\code\infer.py

"""
Inference script for PhosOpt inverse model.

Given:
- a trained inverse model checkpoint (.pt),
- an EMNIST phosphene .npz split,
- and the retinotopy directory for the simulator,

this script:
1) Loads the trained InverseModel and DifferentiableSimulator.
2) Runs inference for every sample in the selected split.
3) Saves per-sample implant parameters, electrode activations, recon maps, and metrics.
4) Saves aggregate statistics for downstream analysis.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Work around duplicate OpenMP runtime initialization on some Windows setups.
# Must be set before importing libraries that load OpenMP (e.g., numpy/torch/scipy).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset import PhospheneDataset  # noqa: E402
from loss.losses import dice_score, hellinger_distance, y_metric_from_params  # noqa: E402
from models.inverse_model import InverseModel  # noqa: E402
from simulator.physics_forward_torch import DifferentiableSimulator  # noqa: E402


# -----------------------------------------------------------------------------
# Inference configuration (edit here instead of CLI args)
# -----------------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "data" / "output" / "inverse_training_v3_halfright" / "single_subject_inverse_model.pt"
RETINOTOPY_DIR = PROJECT_ROOT / "data" / "fmri" / "100610"
HEMISPHERE = "LH"
MAP_SIZE = 128

EMNIST_NPZ = PROJECT_ROOT / "data" / "letters" / "emnist_letters_v3_halfright_128.npz"
EMNIST_SPLIT = "test"

SAVE_ROOT = PROJECT_ROOT / "data" / "output"
RUN_NAME = "infer_v3_halfright"
ELECTRODE_ON_THRESHOLD = 0.5


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
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# def _save_map_png(path: Path, arr: np.ndarray, title: str) -> None:
#     plt.figure(figsize=(5, 5), dpi=150)
#     plt.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
#     plt.title(title)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches="tight", pad_inches=0.02)
#     plt.close()


def main() -> None:
    device = _get_device()
    print(f"[PhosOpt][infer] Using device: {device}")

    # ------------------------------------------------------------------
    # Load dataset split
    # ------------------------------------------------------------------
    if EMNIST_SPLIT not in {"train", "test"}:
        raise ValueError("EMNIST_SPLIT must be 'train' or 'test'")
    key = "train_phosphenes" if EMNIST_SPLIT == "train" else "test_phosphenes"
    dataset = PhospheneDataset.from_phosphene_npz(npz_path=EMNIST_NPZ, key=key)
    n_samples = len(dataset)
    if n_samples == 0:
        raise RuntimeError(f"No samples found in split '{EMNIST_SPLIT}' from {EMNIST_NPZ}")
    print(
        f"[PhosOpt][infer] Loaded split '{EMNIST_SPLIT}' from {EMNIST_NPZ} "
        f"with {n_samples} samples"
    )

    # ------------------------------------------------------------------
    # Load model and simulator
    # ------------------------------------------------------------------
    print(f"[PhosOpt][infer] Loading model from: {MODEL_PATH}")
    model = InverseModel(in_channels=1, latent_dim=256, electrode_dim=1000)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    print(f"[PhosOpt][infer] Initializing DifferentiableSimulator from {RETINOTOPY_DIR}")
    simulator = DifferentiableSimulator(
        data_dir=RETINOTOPY_DIR,
        hemisphere=HEMISPHERE,
        map_size=MAP_SIZE,
    ).to(device).eval()

    # ------------------------------------------------------------------
    # Inference over all split samples
    # ------------------------------------------------------------------
    run_dir = SAVE_ROOT / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | int | dict[str, float]]] = []
    all_dc: list[float] = []
    all_y: list[float] = []
    all_hd: list[float] = []
    all_score: list[float] = []
    all_loss: list[float] = []
    all_num_active_electrodes: list[float] = []
    all_active_ratio: list[float] = []

    with torch.no_grad():
        for idx in range(n_samples):
            target = dataset[idx].unsqueeze(0).to(device)  # [1,1,H,W]
            target_np = target[0, 0].detach().cpu().numpy()

            params, electrode_logits = model(target)
            recon = simulator(params, electrode_logits)

            dc_metric = float(dice_score(recon, target).item())
            hd_metric = float(hellinger_distance(recon, target).item())
            y_metric = float(y_metric_from_params(simulator, params).item())
            score = dc_metric + 0.1 * y_metric - hd_metric
            loss = 2.0 - score

            params_vec = params[0].detach().cpu().numpy().tolist()
            elec_prob = torch.sigmoid(electrode_logits)[0].detach().cpu().numpy()
            recon_np = recon[0, 0].detach().cpu().numpy()
            num_active_electrodes = int((elec_prob > ELECTRODE_ON_THRESHOLD).sum())
            active_ratio = float(num_active_electrodes / elec_prob.size)

            base = run_dir / f"sample_{idx:05d}"
            params_path = base.with_suffix(".params.json")
            elec_path = base.with_suffix(".electrodes.npy")
            recon_path = base.with_suffix(".recon.npy")
            target_path = base.with_suffix(".target.npy")
            # recon_img_path = base.with_suffix(".recon.png")
            # target_img_path = base.with_suffix(".target.png")

            learned_params = {
                "alpha": params_vec[0],
                "beta": params_vec[1],
                "offset_from_base": params_vec[2],
                "shank_length": params_vec[3],
            }
            row = {
                "sample_index": idx,
                "learned_implant_params": learned_params,
                "electrode_stats": {
                    "on_threshold": ELECTRODE_ON_THRESHOLD,
                    "num_active_electrodes": num_active_electrodes,
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
            meta = {
                "model_path": str(MODEL_PATH),
                "retinotopy_dir": str(RETINOTOPY_DIR),
                "hemisphere": HEMISPHERE,
                "map_size": MAP_SIZE,
                "target_source": str(EMNIST_NPZ),
                "emnist_split": EMNIST_SPLIT,
                **row,
            }

            params_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            np.save(elec_path, elec_prob)
            np.save(recon_path, recon_np)
            np.save(target_path, target_np)
            # _save_map_png(recon_img_path, recon_np, title=f"Reconstructed #{idx}")
            # _save_map_png(target_img_path, target_np, title=f"Target #{idx}")

            summary_rows.append(row)
            all_dc.append(dc_metric)
            all_y.append(y_metric)
            all_hd.append(hd_metric)
            all_score.append(score)
            all_loss.append(loss)
            all_num_active_electrodes.append(float(num_active_electrodes))
            all_active_ratio.append(active_ratio)

            if (idx + 1) % 10 == 0 or (idx + 1) == n_samples:
                print(f"[PhosOpt][infer] Processed {idx + 1}/{n_samples} samples")

    summary = {
        "model_path": str(MODEL_PATH),
        "retinotopy_dir": str(RETINOTOPY_DIR),
        "hemisphere": HEMISPHERE,
        "map_size": MAP_SIZE,
        "target_source": str(EMNIST_NPZ),
        "emnist_split": EMNIST_SPLIT,
        "num_samples": n_samples,
        "aggregate_performance": {
            "dc_metric": _stat(all_dc),
            "y_metric": _stat(all_y),
            "hd_metric": _stat(all_hd),
            "score": _stat(all_score),
            "loss": _stat(all_loss),
        },
        "aggregate_electrode_stats": {
            "on_threshold": ELECTRODE_ON_THRESHOLD,
            "num_active_electrodes": _stat(all_num_active_electrodes),
            "active_ratio": _stat(all_active_ratio),
        },
    }

    summary_path = run_dir / "summary.json"
    summary_jsonl_path = run_dir / "per_sample_metrics.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with summary_jsonl_path.open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[PhosOpt][infer] Saved run directory: {run_dir}")
    print(f"[PhosOpt][infer] Saved aggregate summary: {summary_path}")
    print(f"[PhosOpt][infer] Saved per-sample metrics JSONL: {summary_jsonl_path}")
    print(
        "[PhosOpt][infer] Aggregate means -> "
        f"DC={summary['aggregate_performance']['dc_metric']['mean']:.6f}, "
        f"Y={summary['aggregate_performance']['y_metric']['mean']:.6f}, "
        f"HD={summary['aggregate_performance']['hd_metric']['mean']:.6f}, "
        f"Score={summary['aggregate_performance']['score']['mean']:.6f}, "
        f"Loss={summary['aggregate_performance']['loss']['mean']:.6f}, "
        f"ActiveElec={summary['aggregate_electrode_stats']['num_active_electrodes']['mean']:.2f}"
    )


if __name__ == "__main__":
    main()

