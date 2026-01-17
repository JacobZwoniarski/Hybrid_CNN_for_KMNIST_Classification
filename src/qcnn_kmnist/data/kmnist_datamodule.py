"""
KMNIST data loading via torchvision.
Will download automatically to ./data_cache
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class KMNISTLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2,
    val_fraction: float = 0.1,
    seed: int = 42,
    data_dir: str | Path = "data_cache",
    download: bool = True,
) -> KMNISTLoaders:
    """
    Returns train/val/test DataLoaders for KMNIST.

    Notes:
    - Uses torchvision.datasets.KMNIST (downloads if needed).
    - Normalization uses mean=0.5, std=0.5 -> maps [0,1] to [-1,1].
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0,1).")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    full_train = datasets.KMNIST(
        root=str(data_path),
        train=True,
        transform=transform,
        download=download,
    )
    test_ds = datasets.KMNIST(
        root=str(data_path),
        train=False,
        transform=transform,
        download=download,
    )

    n_total = len(full_train)
    n_val = int(round(n_total * val_fraction))
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, lengths=[n_train, n_val], generator=gen)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return KMNISTLoaders(train=train_loader, val=val_loader, test=test_loader)
