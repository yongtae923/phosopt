"""
Pilot: self-contained PhosOpt-style pipeline inside `pilot/`.

- Trains an inverse model from target maps (MNIST-like letters) using a
  simple differentiable simulator defined here.
- Saves the model checkpoint into a results directory.
- Runs zero-shot and per-target fine-tuning on a subset of MNIST-letter
  targets and logs metrics.

This script:
- does NOT import anything from the top-level `code/` package
- only reads from `data/` and writes under a user-specified results dir.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_ELECTRODES = 1000


# ---------------------------------------------------------------------------
# Data utilities (MNIST-like letter maps from data/letters)
# ---------------------------------------------------------------------------

def _normalise_map(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = m.astype(np.float32)
    mx = float(arr.max())
    if mx > eps:
        arr = arr / mx
    s = float(arr.sum())
    if s > eps:
        arr = arr / s
    return arr.astype(np.float32)


def _load_emnist_letters_from_npz(
    map_size: int,
    max_maps: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Load letter-like phosphene maps from data/letters/*.npz.
    """
    letters_dir = PROJECT_ROOT / "data" / "letters"
    candidates = [
        letters_dir / "emnist_letters_five.npz",
        letters_dir / "emnist_letters_phosphenes.npz",
    ]
    npz_path = next((p for p in candidates if p.exists()), None)
    if npz_path is None:
        raise FileNotFoundError(
            "Could not find EMNIST letters npz in data/letters/. "
            "Expected one of: emnist_letters_five.npz, emnist_letters_phosphenes.npz"
        )

    with np.load(npz_path, allow_pickle=False) as f:
        if "phosphenes" in f.files:
            raw = f["phosphenes"].astype(np.float32)
        else:
            raw = f[f.files[0]].astype(np.float32)

    if raw.ndim == 4 and raw.shape[1] == 1:
        raw = raw[:, 0, :, :]

    n_total = min(max_maps, raw.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(raw.shape[0], size=n_total, replace=False)
    raw = raw[idx]

    maps: Dict[str, np.ndarray] = {}
    for i, m in enumerate(raw):
        if m.shape[0] != map_size or m.shape[1] != map_size:
            t = torch.from_numpy(m[None, None, ...])
            t_resized = torch.nn.functional.interpolate(
                t, size=(map_size, map_size), mode="bilinear", align_corners=False
            )
            arr = t_resized.squeeze(0).squeeze(0).cpu().numpy()
        else:
            arr = m
        maps[f"mnist_letter_{i:02d}"] = _normalise_map(arr)
    return maps


def load_all_targets(
    map_size: int = 256,
    seed: int = 0,
    max_maps: int = 20,
) -> Dict[str, np.ndarray]:
    """Return dict[name -> map] of MNIST-letter-style targets."""
    return _load_emnist_letters_from_npz(map_size=map_size, max_maps=max_maps, seed=seed)


def select_mnist_subset(
    targets: Dict[str, np.ndarray],
    n: int,
) -> Dict[str, np.ndarray]:
    keys = sorted([k for k in targets.keys() if k.startswith("mnist_letter_")])
    keys = keys[: max(1, n)]
    return {k: targets[k] for k in keys}


# ---------------------------------------------------------------------------
# Simple differentiable simulator (params + electrode logits -> map)
# ---------------------------------------------------------------------------

class SimpleSimulator(nn.Module):
    """
    Tiny differentiable simulator:

    - takes 4 stimulation parameters + 1000 electrode logits
    - applies sigmoid to logits (electrode activations)
    - linearly projects activations into a [H,W] map
    - uses the 4 params to scale/shift intensity
    """

    def __init__(self, map_size: int = 256, n_electrodes: int = N_ELECTRODES) -> None:
        super().__init__()
        self.map_size = map_size
        self.n_electrodes = n_electrodes
        self.linear = nn.Linear(n_electrodes, map_size * map_size)

    def forward(self, params: torch.Tensor, electrode_logits: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(electrode_logits)  # [B, E]
        img = self.linear(x).view(-1, 1, self.map_size, self.map_size)

        alpha, beta, offset, shank = torch.unbind(params, dim=1)
        alpha = alpha.view(-1, 1, 1, 1)
        offset = offset.view(-1, 1, 1, 1)
        img = alpha * img + offset
        img = torch.relu(img)

        mx = img.amax(dim=(1, 2, 3), keepdim=True)
        img = img / (mx + 1e-6)
        return img


# ---------------------------------------------------------------------------
# Inverse model (encoder + heads)
# ---------------------------------------------------------------------------

class InverseModel(nn.Module):
    """Small CNN encoder that outputs 4 params + 1000 electrode logits."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 128, electrode_dim: int = N_ELECTRODES) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.param_head = nn.Linear(latent_dim, 4)
        self.electrode_head = nn.Linear(latent_dim, electrode_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.fc(h)
        params = self.param_head(z)
        electrode_logits = self.electrode_head(z)
        return params, electrode_logits


# ---------------------------------------------------------------------------
# Losses and metrics
# ---------------------------------------------------------------------------

@dataclass
class LossConfig:
    recon_weight: float = 1.0
    sparsity_weight: float = 1e-3
    param_prior_weight: float = 1e-4


def compute_losses(
    recon: torch.Tensor,
    target: torch.Tensor,
    params: torch.Tensor,
    electrode_logits: torch.Tensor,
    cfg: LossConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mse = torch.mean((recon - target) ** 2)
    sparsity = torch.mean(torch.sigmoid(electrode_logits))
    param_l2 = torch.mean(params ** 2)

    loss = (
        cfg.recon_weight * mse
        + cfg.sparsity_weight * sparsity
        + cfg.param_prior_weight * param_l2
    )
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "mse": float(mse.detach().cpu().item()),
    }
    return loss, metrics


def evaluate_score(recon_np: np.ndarray, target_np: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((recon_np - target_np) ** 2))
    score = 2.0 - mse
    return {"score": score, "loss": mse}


# ---------------------------------------------------------------------------
# Training: shared encoder PhosOpt-style model
# ---------------------------------------------------------------------------

def train_phosopt(
    targets: Dict[str, np.ndarray],
    simulator: SimpleSimulator,
    seed: int,
    output_dir: Path,
    max_epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
) -> Path:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    maps = np.stack(list(targets.values()), axis=0)
    t = torch.from_numpy(maps).unsqueeze(1).float()
    n = t.shape[0]
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15))
    idx = torch.randperm(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]

    train_ds = TensorDataset(t[train_idx])
    val_ds = TensorDataset(t[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = InverseModel(in_channels=1, latent_dim=128, electrode_dim=N_ELECTRODES).to(device)
    simulator = simulator.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_cfg = LossConfig()

    best_val = float("inf")
    best_state = None
    wait = 0

    for _epoch in range(max_epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            params, logits = model(batch)
            recon = simulator(params, logits)
            loss, _ = compute_losses(recon, batch, params, logits, loss_cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                params, logits = model(batch)
                recon = simulator(params, logits)
                loss, _ = compute_losses(recon, batch, params, logits, loss_cfg)
                val_loss += float(loss.detach().cpu().item())
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"phosopt_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def load_model(checkpoint_path: Path) -> InverseModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InverseModel(in_channels=1, latent_dim=128, electrode_dim=N_ELECTRODES).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-target optimisation on MNIST letters
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    max_simulator_calls: int = 100


def phosopt_zeroshot(
    model: InverseModel,
    simulator: SimpleSimulator,
    target_map: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    t = torch.from_numpy(target_map).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        params, logits = model(t)
        recon = simulator(params, logits)
        recon_np = recon.squeeze(0).squeeze(0).cpu().numpy()
    metrics = evaluate_score(recon_np, target_map)
    active = int((torch.sigmoid(logits).squeeze().cpu().numpy() > 0.5).sum())
    return {
        "benchmark_type": "pilot_mnist",
        "method": "phosopt_zeroshot",
        "seed": seed,
        "score": metrics["score"],
        "loss": metrics["loss"],
        "active_electrode_count": active,
        "simulator_calls": 1,
        "wall_clock_time": 0.0,
    }


def phosopt_finetune(
    model: InverseModel,
    simulator: SimpleSimulator,
    target_map: np.ndarray,
    budget: BudgetConfig,
    seed: int,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    ft_model = InverseModel(in_channels=1, latent_dim=128, electrode_dim=N_ELECTRODES).to(device)
    ft_model.load_state_dict(model.state_dict())
    ft_model.train()

    target_t = torch.from_numpy(target_map).unsqueeze(0).unsqueeze(0).float().to(device)
    opt = torch.optim.Adam(ft_model.parameters(), lr=lr)
    loss_cfg = LossConfig()

    best_score = -float("inf")
    best_metrics: Dict[str, float] = {}
    call_count = 0

    while call_count < budget.max_simulator_calls:
        params, logits = ft_model(target_t)
        recon = simulator(params, logits)
        call_count += 1

        loss, _ = compute_losses(recon, target_t, params, logits, loss_cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            recon_np = recon.squeeze(0).squeeze(0).cpu().numpy()
            metrics = evaluate_score(recon_np, target_map)
            if metrics["score"] > best_score:
                best_score = metrics["score"]
                best_metrics = metrics

    ft_model.eval()
    with torch.no_grad():
        _, logits = ft_model(target_t)
    active = int((torch.sigmoid(logits).squeeze().cpu().numpy() > 0.5).sum())

    return {
        "benchmark_type": "pilot_mnist",
        "method": "phosopt_finetune",
        "seed": seed,
        "score": best_metrics.get("score", best_score),
        "loss": best_metrics.get("loss", float("nan")),
        "active_electrode_count": active,
        "simulator_calls": call_count,
        "wall_clock_time": 0.0,
    }


def evaluate_and_optimize_mnist(
    mnist_targets: Dict[str, np.ndarray],
    model: InverseModel,
    simulator: SimpleSimulator,
    seed: int,
    max_sim_calls: int,
) -> List[Dict[str, Any]]:
    budget = BudgetConfig(max_simulator_calls=max_sim_calls)
    rows: List[Dict[str, Any]] = []
    for tid, tmap in mnist_targets.items():
        log_zs = phosopt_zeroshot(model, simulator, tmap, seed)
        log_zs["target_id"] = tid
        rows.append(log_zs)

        log_ft = phosopt_finetune(model, simulator, tmap, budget, seed)
        log_ft["target_id"] = tid
        rows.append(log_ft)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    methods = sorted(set(r["method"] for r in rows))
    out: Dict[str, Any] = {"methods": {}}
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        if not sub:
            continue
        out["methods"][m] = {
            "n": len(sub),
            "score_mean": float(np.mean([r["score"] for r in sub])),
            "loss_mean": float(np.mean([r["loss"] for r in sub])),
            "active_electrode_count_mean": float(np.mean([r["active_electrode_count"] for r in sub])),
            "simulator_calls_mean": float(np.mean([r["simulator_calls"] for r in sub])),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Pilot PhosOpt (self-contained in pilot/)")
    p.add_argument("--output_dir", type=Path, default=Path("pilot/results"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--map_size", type=int, default=256)

    # Train config
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=8)

    # MNIST optimization config
    p.add_argument("--mnist_n", type=int, default=5, help="Number of MNIST letter targets to optimize")
    p.add_argument("--max_sim_calls", type=int, default=100)

    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=== Pilot (self-contained) ===")
    print(f"Output dir: {out}")
    print(f"Seed: {args.seed}")

    # 1) Load targets from data/letters
    all_targets = load_all_targets(map_size=args.map_size, seed=args.seed, max_maps=max(args.mnist_n, 10))
    mnist_targets = select_mnist_subset(all_targets, n=args.mnist_n)

    # 2) Build simulator + train inverse model
    simulator = SimpleSimulator(map_size=args.map_size, n_electrodes=N_ELECTRODES)
    ckpt_path = train_phosopt(
        targets=all_targets,
        simulator=simulator,
        seed=args.seed,
        output_dir=out,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    print(f"Saved model checkpoint: {ckpt_path}")

    # 3) Load model and evaluate MNIST subset
    model = load_model(ckpt_path)
    rows = evaluate_and_optimize_mnist(
        mnist_targets=mnist_targets,
        model=model,
        simulator=simulator,
        seed=args.seed,
        max_sim_calls=args.max_sim_calls,
    )

    # 4) Save logs and summary under output_dir
    jsonl_path = out / "pilot_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote results: {jsonl_path} ({len(rows)} rows)")

    summary = summarize(rows)
    summary_path = out / "pilot_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()

