"""
Experiment orchestrator for PhosOpt.

Reads a YAML config file and dispatches to the appropriate benchmark
runner (per_target, generalized, adaptation).

Usage:
    python code/experiment.py --config configs/pilot.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def _load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_simulator(cfg: dict):
    """Instantiate the simulator specified in the config."""
    sim_cfg = cfg.get("simulator", {})
    backend = sim_cfg.get("backend", "numpy")
    retino_dir = PROJECT_ROOT / sim_cfg.get("retinotopy_dir", "data/fmri/100610")
    hemisphere = sim_cfg.get("hemisphere", "LH")
    map_size = sim_cfg.get("map_size", 256)

    if backend == "diff":
        from simulator.physics_forward_torch import DifferentiableSimulator
        import torch
        sim = DifferentiableSimulator(
            data_dir=retino_dir, hemisphere=hemisphere, map_size=map_size,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return sim.to(device)
    else:
        from simulator.simulator_wrapper import NumpySimulatorAdapter
        return NumpySimulatorAdapter(data_dir=retino_dir, hemisphere=hemisphere)


def _load_targets(cfg: dict) -> dict:
    """Load or generate target maps according to config."""
    import numpy as np
    from targets.generator import generate_all_targets, load_targets

    target_dir = PROJECT_ROOT / "data" / "targets"
    if target_dir.exists() and any(target_dir.glob("*.npy")):
        all_targets = load_targets(target_dir)
    else:
        npz_path = PROJECT_ROOT / "data" / "letters" / "emnist_letters_five.npz"
        npz = npz_path if npz_path.exists() else PROJECT_ROOT / "data" / "letters" / "emnist_letters_phosphenes.npz"
        npz = npz if npz.exists() else None
        all_maps, category_suffixes = generate_all_targets(
            size=cfg.get("simulator", {}).get("map_size", 256),
            npz_path=npz,
        )
        from targets.generator import save_targets
        save_targets(all_maps, target_dir, category_suffixes=category_suffixes)
        all_targets = load_targets(target_dir)

    # Filter by config
    tcfg = cfg.get("targets", {})
    categories = tcfg.get("categories", None)
    indices_spec = tcfg.get("indices", "all")

    if categories:
        filtered = {}
        for key, val in all_targets.items():
            cat = key.rsplit("_", 1)[0]
            if cat in categories:
                if indices_spec == "all":
                    filtered[key] = val
                elif isinstance(indices_spec, list):
                    idx_str = key.rsplit("_", 1)[-1]
                    try:
                        idx_int = int(idx_str)
                    except ValueError:
                        continue
                    if idx_int in indices_spec:
                        filtered[key] = val
        return filtered

    return all_targets


def run_per_target(cfg: dict) -> None:
    from baselines.base import BudgetConfig
    from benchmarks.per_target import run_per_target_benchmark

    budget_cfg = cfg.get("budget", {})
    es_cfg = cfg.get("early_stopping", {})
    budget = BudgetConfig(
        max_simulator_calls=budget_cfg.get("max_simulator_calls", 300),
        max_wall_clock_sec=budget_cfg.get("max_wall_clock_min", 20) * 60,
        patience_calls=es_cfg.get("patience_calls", 30),
        min_improvement=es_cfg.get("min_improvement", 1e-4),
    )

    targets = _load_targets(cfg)
    simulator = _build_simulator(cfg)
    methods = cfg.get("methods", ["bayesian"])
    seeds = cfg.get("seeds", [42])
    output_dir = PROJECT_ROOT / cfg.get("output_dir", "results/per_target")
    sim_cfg = cfg.get("simulator", {})
    data_dir = PROJECT_ROOT / sim_cfg.get("retinotopy_dir", "data/fmri/100610")
    hemisphere = sim_cfg.get("hemisphere", "LH")

    print(f"=== Per-target Benchmark ===")
    print(f"  Targets: {len(targets)}")
    print(f"  Methods: {methods}")
    print(f"  Seeds: {seeds}")
    print(f"  Budget: {budget.max_simulator_calls} calls, "
          f"{budget.max_wall_clock_sec / 60:.0f} min")

    run_per_target_benchmark(
        targets=targets,
        simulator_fn=simulator,
        method_names=methods,
        seeds=seeds,
        budget=budget,
        output_dir=output_dir,
        data_dir=data_dir,
        hemisphere=hemisphere,
    )


def run_generalized(cfg: dict) -> None:
    from benchmarks.generalized import run_generalized_benchmark

    targets = _load_targets(cfg)
    simulator = _build_simulator(cfg)
    methods = cfg.get("methods", ["phosopt"])
    seeds = cfg.get("seeds", [42])
    output_dir = PROJECT_ROOT / cfg.get("output_dir", "results/generalized")
    train_cfg = cfg.get("training", {})

    print(f"=== Generalized Benchmark ===")
    print(f"  Targets: {len(targets)}")
    print(f"  Methods: {methods}")
    print(f"  Seeds: {seeds}")

    run_generalized_benchmark(
        targets=targets,
        simulator=simulator,
        method_names=methods,
        seeds=seeds,
        output_dir=output_dir,
        max_epochs=train_cfg.get("max_epochs", 100),
        patience=train_cfg.get("patience", 10),
        lr=train_cfg.get("lr", 1e-3),
        shared_params_lr=train_cfg.get("shared_params_lr", 1e-2),
        batch_size=train_cfg.get("batch_size", 8),
        train_ratio=cfg.get("targets", {}).get("train_ratio", 0.7),
        val_ratio=cfg.get("targets", {}).get("val_ratio", 0.15),
    )


def run_adaptation(cfg: dict) -> None:
    import torch
    from baselines.base import BudgetConfig
    from benchmarks.adaptation import run_adaptation_benchmark
    from benchmarks.generalized import _split_targets
    from models.inverse_model import InverseModel

    budget_cfg = cfg.get("budget", {})
    es_cfg = cfg.get("early_stopping", {})
    budget = BudgetConfig(
        max_simulator_calls=budget_cfg.get("max_simulator_calls", 100),
        max_wall_clock_sec=budget_cfg.get("max_wall_clock_min", 10) * 60,
        patience_calls=es_cfg.get("patience_calls", 20),
        min_improvement=es_cfg.get("min_improvement", 1e-4),
    )

    all_targets = _load_targets(cfg)
    simulator = _build_simulator(cfg)
    seeds = cfg.get("seeds", [42])
    output_dir = PROJECT_ROOT / cfg.get("output_dir", "results/adaptation")

    # Apply test split so adaptation only uses unseen targets
    tcfg = cfg.get("targets", {})
    split = tcfg.get("split", None)
    if split:
        train_ratio = tcfg.get("train_ratio", 0.7)
        val_ratio = tcfg.get("val_ratio", 0.15)
        split_seed = tcfg.get("split_seed", 42)
        train_maps, val_maps, test_maps = _split_targets(
            all_targets, train_ratio, val_ratio, split_seed,
        )
        if split == "test":
            targets = test_maps
        elif split == "val":
            targets = val_maps
        elif split == "train":
            targets = train_maps
        else:
            targets = all_targets
    else:
        targets = all_targets

    # Load pre-trained model
    model_path = PROJECT_ROOT / cfg.get("pretrained_model", "results/generalized/best_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InverseModel(in_channels=1, latent_dim=256, electrode_dim=1000)

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded pre-trained model: {model_path}")
    else:
        print(f"[WARN] Pre-trained model not found at {model_path}, using random init.")
    model = model.to(device)

    sim_cfg = cfg.get("simulator", {})
    data_dir = PROJECT_ROOT / sim_cfg.get("retinotopy_dir", "data/fmri/100610")
    hemisphere = sim_cfg.get("hemisphere", "LH")

    print(f"=== Adaptation Benchmark ===")
    print(f"  Test targets: {len(targets)}")
    print(f"  Seeds: {seeds}")
    print(f"  Budget: {budget.max_simulator_calls} calls")

    run_adaptation_benchmark(
        test_targets=targets,
        pretrained_model=model,
        simulator=simulator,
        seeds=seeds,
        budget=budget,
        output_dir=output_dir,
        data_dir=data_dir,
        hemisphere=hemisphere,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PhosOpt experiment orchestrator")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    benchmark = cfg.get("benchmark", "per_target")

    print(f"PhosOpt Experiment: benchmark={benchmark}")
    print(f"Config: {args.config}")

    if benchmark == "per_target":
        run_per_target(cfg)
    elif benchmark == "generalized":
        run_generalized(cfg)
    elif benchmark == "adaptation":
        run_adaptation(cfg)
    else:
        print(f"Unknown benchmark type: {benchmark}")
        sys.exit(1)


if __name__ == "__main__":
    main()
