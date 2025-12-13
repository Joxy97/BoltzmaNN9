"""Dataset and DataLoader utilities for Boltzmann Machines."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class BMDataset(Dataset):
    """PyTorch Dataset for Boltzmann Machine training data.

    Loads data from a CSV file.

    Args:
        csv_path: Path to CSV file containing the training data.
            Each row is a sample, columns are features.
        drop_cols: Optional list of column names to drop from the data.
    """

    def __init__(
        self,
        csv_path: str | Path,
        drop_cols: Sequence[str] | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        if drop_cols:
            self.df = self.df.drop(columns=list(drop_cols))

        self.columns = list(self.df.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row.values, dtype=torch.float32)
        return x


class GBMDataloader:
    """Simple wrapper around PyTorch DataLoader for Boltzmann Machine data.

    Args:
        dataset: BMDataset instance.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data each epoch.
        num_workers: Number of worker processes for data loading.
    """

    def __init__(
        self,
        dataset: BMDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def split_rbm_loaders(
    dataset: BMDataset,
    *,
    batch_size: int,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last_train: bool = True,
) -> Dict[str, DataLoader]:
    """Split dataset and create train/val/test DataLoaders.

    Args:
        dataset: BMDataset to split.
        batch_size: Number of samples per batch.
        split: Tuple of (train_frac, val_frac, test_frac). Must sum to 1.0.
        seed: Random seed for reproducible splits.
        shuffle_train: Whether to shuffle training data.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory (good for CUDA).
        drop_last_train: Whether to drop last incomplete batch during training.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoader instances.

    Raises:
        ValueError: If split fractions don't sum to 1.0.
    """
    train_frac, val_frac, test_frac = split
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1.0")

    n = len(dataset)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=gen
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=drop_last_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
