"""
Generalized benchmark runner (exp.md Section 3.2).

A single **shared parameter** model is trained on a training set and
evaluated on a held-out test set *without* any additional per-target
optimisation at test time.

Methods: all_on_shared, random_shared, heuristic_shared, phosopt

Shared baselines learn one set of 4 stimulation parameters with a fixed
electrode mask strategy; PhosOpt learns an encoder that outputs per-image
parameters and electrode logits.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from logger import ExperimentLog, ExperimentLogger
from metrics.eval_metrics import evaluate_all


# ---------------------------------------------------------------------------
# Train/val/test split (also imported by experiment.py for adaptation)
# ---------------------------------------------------------------------------

def _split_targets(
    targets: dict[str, np.ndarray],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split target dict into train / val / test."""
    keys = sorted(targets.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    train_keys = keys[:n_train]
    val_keys = keys[n_train: n_train + n_val]
    test_keys = keys[n_train + n_val:]
    if not test_keys:
        test_keys = val_keys[-1:]
    return (
        {k: targets[k] for k in train_keys},
        {k: targets[k] for k in val_keys},
        {k: targets[k] for k in test_keys},
    )


# ---------------------------------------------------------------------------
# Electrode mask strategies for shared baselines
# ---------------------------------------------------------------------------

def _make_shared_mask(
    method: str,
    train_maps: dict[str, np.ndarray],
    n_electrodes: int = 1000,
    active_ratio: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Return a fixed electrode mask for a shared baseline strategy."""
    if method == "all_on_shared":
        return np.ones(n_electrodes, dtype=np.float32)

    if method == "random_shared":
        rng = np.random.default_rng(seed)
        n_active = max(1, int(n_electrodes * active_ratio))
        idx = rng.choice(n_electrodes, size=n_active, replace=False)
        mask = np.zeros(n_electrodes, dtype=np.float32)
        mask[idx] = 1.0
        return mask

    if method == "heuristic_shared":
        avg_map = np.mean(np.stack(list(train_maps.values())), axis=0)
        from baselines.heuristic_subset import _intensity_prior_mask
        return _intensity_prior_mask(avg_map, n_electrodes, active_ratio)

    raise ValueError(f"Unknown shared mask method: {method}")


# ---------------------------------------------------------------------------
# Shared baseline training (4 params + fixed mask)
# ---------------------------------------------------------------------------

def _train_shared_baseline(
    train_maps: dict[str, np.ndarray],
    val_maps: dict[str, np.ndarray],
    simulator: Any,
    electrode_mask: np.ndarray,
    max_epochs: int,
    patience: int,
    lr: float,
    seed: int,
    method_name: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Optimise 4 shared stimulation params with a fixed electrode mask.

    Returns (best_params_np, electrode_mask, training_wall_time).
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shared_params = torch.nn.Parameter(
        torch.tensor([0.0, 0.0, 20.0, 25.0], dtype=torch.float32, device=device)
    )
    logits_val = np.where(electrode_mask > 0.5, 8.0, -8.0).astype(np.float32)
    fixed_logits = torch.tensor(logits_val, device=device).unsqueeze(0)

    opt = torch.optim.Adam([shared_params], lr=lr)

    train_ts = [
        torch.from_numpy(m).unsqueeze(0).unsqueeze(0).float().to(device)
        for m in train_maps.values()
    ]
    val_ts = [
        (tid, torch.from_numpy(m).unsqueeze(0).unsqueeze(0).float().to(device))
        for tid, m in val_maps.items()
    ]

    best_val_score = -float("inf")
    best_params_np = shared_params.detach().cpu().numpy().copy()
    wait = 0
    t0 = time.perf_counter()

    for epoch in range(max_epochs):
        # --- train ---
        for t_target in train_ts:
            params_batch = shared_params.unsqueeze(0)
            recon = simulator(params_batch, fixed_logits)
            loss = torch.mean((recon - t_target) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # --- val ---
        val_score = 0.0
        with torch.no_grad():
            params_batch = shared_params.unsqueeze(0)
            for _, vt in val_ts:
                recon = simulator(params_batch, fixed_logits)
                recon_np = recon.squeeze().cpu().numpy()
                target_np = vt.squeeze().cpu().numpy()
                m = evaluate_all(recon_np, target_np)
                val_score += m["score"]
        val_score /= max(len(val_ts), 1)

        if val_score > best_val_score + 1e-4:
            best_val_score = val_score
            best_params_np = shared_params.detach().cpu().numpy().copy()
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    training_time = time.perf_counter() - t0
    return best_params_np, electrode_mask, training_time


def _evaluate_shared_on_targets(
    shared_params_np: np.ndarray,
    electrode_mask: np.ndarray,
    simulator: Any,
    test_maps: dict[str, np.ndarray],
) -> list[tuple[str, dict[str, float], int]]:
    """Evaluate fixed shared params + mask on each test target."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(simulator, "parameters"):
        try:
            device = next(simulator.parameters()).device
        except StopIteration:
            pass

    params_t = torch.from_numpy(shared_params_np.astype(np.float32)).unsqueeze(0).to(device)
    logits_val = np.where(electrode_mask > 0.5, 8.0, -8.0).astype(np.float32)
    fixed_logits = torch.tensor(logits_val, device=device).unsqueeze(0)
    active = int((electrode_mask > 0.5).sum())

    results: list[tuple[str, dict[str, float], int]] = []
    with torch.no_grad():
        for tid, tmap in test_maps.items():
            recon = simulator(params_t, fixed_logits)
            recon_np = recon.squeeze().cpu().numpy()
            metrics = evaluate_all(recon_np, tmap)
            results.append((tid, metrics, active))
    return results


# ---------------------------------------------------------------------------
# PhosOpt (InverseModel) training
# ---------------------------------------------------------------------------

def _train_phosopt(
    train_maps: dict[str, np.ndarray],
    val_maps: dict[str, np.ndarray],
    simulator: Any,
    max_epochs: int,
    patience: int,
    lr: float,
    shared_params_lr: float,
    batch_size: int,
    seed: int,
    save_dir: Path,
) -> tuple[Any, float]:
    """Train the PhosOpt InverseModel using the existing training pipeline."""
    from models.inverse_model import InverseModel
    from loss.losses import LossConfig, build_losses, dice_score
    from trainer import TrainConfig, train_inverse_model, evaluate_inverse_model

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_loader(maps: dict[str, np.ndarray], shuffle: bool) -> DataLoader:
        arrs = np.stack(list(maps.values()), axis=0)
        t = torch.from_numpy(arrs).unsqueeze(1).float()
        return DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=shuffle)

    train_loader = _to_loader(train_maps, shuffle=True)
    val_loader = _to_loader(val_maps, shuffle=False)

    class _MapLoader:
        def __init__(self, dl: DataLoader) -> None:
            self._dl = dl
        def __iter__(self):
            for (batch,) in self._dl:
                yield batch
        def __len__(self) -> int:
            return len(self._dl)

    model = InverseModel(in_channels=1, latent_dim=256, electrode_dim=1000).to(device)
    simulator = simulator.to(device)

    loss_config = LossConfig(
        recon_weight=1.0, dice_weight=0.5,
        sparsity_weight=1e-3, param_prior_weight=1e-4,
        invalid_region_weight=1e-3, warmup_epochs=5,
    )
    train_config = TrainConfig(
        epochs=max_epochs, batch_size=batch_size,
        lr=lr, shared_params_lr=shared_params_lr,
        weight_decay=1e-4, grad_clip_norm=1.0,
        scheduler_patience=patience,
    )

    t0 = time.perf_counter()
    checkpoint_dir = save_dir / f"seed{seed}_checkpoints"
    train_inverse_model(
        model=model,
        simulator=simulator,
        train_loader=_MapLoader(train_loader),
        val_loader=_MapLoader(val_loader),
        loss_config=loss_config,
        train_config=train_config,
        checkpoint_dir=checkpoint_dir,
    )
    training_time = time.perf_counter() - t0

    model_path = save_dir / f"phosopt_seed{seed}.pt"
    torch.save(model.state_dict(), model_path)

    return model, training_time


def _evaluate_model_on_targets(
    model: Any,
    simulator: Any,
    test_maps: dict[str, np.ndarray],
) -> list[tuple[str, dict[str, float], int]]:
    """Run zero-shot inference and compute metrics per target."""
    device = next(model.parameters()).device
    results: list[tuple[str, dict[str, float], int]] = []

    model.eval()
    with torch.no_grad():
        for tid, tmap in test_maps.items():
            t = torch.from_numpy(tmap).unsqueeze(0).unsqueeze(0).float().to(device)
            params, electrode_logits = model(t)
            recon = simulator(params, electrode_logits)
            recon_np = recon.squeeze().cpu().numpy()
            metrics = evaluate_all(recon_np, tmap)
            active = int((torch.sigmoid(electrode_logits).squeeze().cpu().numpy() > 0.5).sum())
            results.append((tid, metrics, active))

    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_generalized_benchmark(
    targets: dict[str, np.ndarray],
    simulator: Any,
    method_names: list[str],
    seeds: list[int],
    output_dir: str | Path,
    max_epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    shared_params_lr: float = 1e-2,
    batch_size: int = 8,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    experiment_id: str = "generalized",
) -> list[ExperimentLog]:
    """Run the generalized benchmark for all requested methods/seeds."""

    out = Path(output_dir)
    logger = ExperimentLogger(out / "generalized_results.jsonl")
    all_logs: list[ExperimentLog] = []

    for seed in seeds:
        train_maps, val_maps, test_maps = _split_targets(
            targets, train_ratio, val_ratio, seed,
        )
        print(f"\n[Generalized] seed={seed}  train={len(train_maps)} "
              f"val={len(val_maps)} test={len(test_maps)}")

        for method in method_names:
            print(f"  Method: {method}")

            if method == "phosopt":
                model, train_time = _train_phosopt(
                    train_maps, val_maps, simulator,
                    max_epochs, patience, lr, shared_params_lr,
                    batch_size, seed, out,
                )
                eval_results = _evaluate_model_on_targets(model, simulator, test_maps)
            else:
                mask = _make_shared_mask(method, train_maps, seed=seed)
                best_params, mask, train_time = _train_shared_baseline(
                    train_maps, val_maps, simulator, mask,
                    max_epochs, patience, shared_params_lr, seed, method,
                )
                eval_results = _evaluate_shared_on_targets(
                    best_params, mask, simulator, test_maps,
                )

            for tid, metrics, active_count in eval_results:
                log = ExperimentLog(
                    experiment_id=experiment_id,
                    benchmark_type="generalized",
                    method=method,
                    seed=seed,
                    target_id=tid,
                    score=metrics["score"],
                    loss=metrics["loss"],
                    dc=metrics["dc"],
                    y=metrics["y"],
                    hd=metrics["hd"],
                    active_electrode_count=active_count,
                    simulator_calls=0,
                    wall_clock_time=0.0,
                    training_cost=train_time,
                    solving_cost=0.0,
                )
                logger.log(log)
                all_logs.append(log)
                print(f"    {tid}: score={metrics['score']:.4f}")

    print(f"\nGeneralized benchmark complete: {len(all_logs)} entries logged.")
    return all_logs
