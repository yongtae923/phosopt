from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        if min(self.train_ratio, self.val_ratio, self.test_ratio) <= 0:
            raise ValueError("All split ratios must be > 0")


def normalize_target_map(target_map: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize map with max-scaling to [0, 1] range."""
    arr = np.asarray(target_map, dtype=np.float32).copy()
    max_val = float(arr.max())
    if max_val > eps:
        arr /= max_val
    return arr


class PhospheneDataset(Dataset):
    """
    Dataset for self-supervised inverse learning.
    Returns normalized map tensor with shape [1, H, W].
    """

    def __init__(self, target_maps: np.ndarray | Iterable[np.ndarray]) -> None:
        if isinstance(target_maps, np.ndarray):
            maps = target_maps
        else:
            maps = np.asarray(list(target_maps))

        # Accept [N,H,W] or [N,1,H,W] inputs.
        if maps.ndim == 4 and maps.shape[1] == 1:
            maps = maps[:, 0, :, :]
        if maps.ndim != 3:
            raise ValueError(f"target_maps must have shape [N,H,W] or [N,1,H,W], got {maps.shape}")

        normalized = np.stack([normalize_target_map(m) for m in maps], axis=0)
        self.maps = normalized.astype(np.float32)

    @classmethod
    def from_npy_dir(cls, npy_dir: str | Path) -> "PhospheneDataset":
        npy_dir = Path(npy_dir)
        npy_files = sorted(npy_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {npy_dir}")
        maps = [np.load(p).astype(np.float32) for p in npy_files]
        return cls(target_maps=np.stack(maps, axis=0))

    @classmethod
    def from_phosphene_npz(cls, npz_path: str | Path, key: str) -> "PhospheneDataset":
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"npz file not found: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(f"Key '{key}' not found in {npz_path}. Available: {data.files}")
            maps = data[key].astype(np.float32)
        return cls(target_maps=maps)

    def __len__(self) -> int:
        return int(self.maps.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.maps[idx]
        return torch.from_numpy(x).unsqueeze(0)


def make_splits(dataset: Dataset, config: SplitConfig) -> tuple[Subset, Subset, Subset]:
    """Create deterministic train/val/test subsets for single-subject protocol."""
    config.validate()

    n = len(dataset)
    idx = np.arange(n)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(idx)

    n_train = int(round(n * config.train_ratio))
    n_val = int(round(n * config.val_ratio))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("Not enough samples to create non-empty splits")

    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train : n_train + n_val].tolist()
    test_idx = idx[n_train + n_val :].tolist()
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def load_letters_phosphene_splits(
    npz_path: str | Path,
    seed: int = 42,
    val_ratio_from_train: float = 0.1,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[Subset, Subset, Subset]:
    """
    Use pre-defined EMNIST phosphene split:
      - train: train_phosphenes
      - test: test_phosphenes
      - val: split from train set
    """
    if not (0.0 < val_ratio_from_train < 1.0):
        raise ValueError("val_ratio_from_train must be in (0, 1)")

    train_ds = PhospheneDataset.from_phosphene_npz(npz_path=npz_path, key="train_phosphenes")
    test_ds = PhospheneDataset.from_phosphene_npz(npz_path=npz_path, key="test_phosphenes")

    train_idx = np.arange(len(train_ds))
    rng = np.random.default_rng(seed)
    rng.shuffle(train_idx)
    n_val = int(round(len(train_idx) * val_ratio_from_train))
    n_val = max(1, min(n_val, len(train_idx) - 1))

    val_indices = train_idx[:n_val].tolist()
    train_indices = train_idx[n_val:].tolist()

    if max_train_samples is not None:
        train_indices = train_indices[: max(1, int(max_train_samples))]
    if max_val_samples is not None:
        val_indices = val_indices[: max(1, int(max_val_samples))]
    if max_test_samples is not None:
        test_indices = list(range(min(len(test_ds), max(1, int(max_test_samples)))))
        test_subset = Subset(test_ds, test_indices)
    else:
        test_subset = Subset(test_ds, list(range(len(test_ds))))

    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(train_ds, val_indices)
    return train_subset, val_subset, test_subset
