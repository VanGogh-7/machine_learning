from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor


def build_transform() -> Compose:
    return Compose([ToTensor()])


def build_datasets(
    data_root: Path,
    valid_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1.")

    transform = build_transform()
    train_valid_dataset = FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = FashionMNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    valid_size = int(len(train_valid_dataset) * valid_ratio)
    train_size = len(train_valid_dataset) - valid_size
    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, valid_dataset = random_split(
        train_valid_dataset,
        [train_size, valid_size],
        generator=split_generator,
    )
    return train_dataset, valid_dataset, test_dataset


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    valid_ratio: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, valid_dataset, test_dataset = build_datasets(
        data_root=data_root,
        valid_ratio=valid_ratio,
        seed=seed,
    )
    loader_generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        **loader_options,
    )
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return train_loader, valid_loader, test_loader

