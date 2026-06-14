from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


def build_transform(
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Callable:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_datasets(
    data_root: Path,
    seed: int,
    train_size: int,
    valid_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> tuple[Dataset, Dataset, Dataset]:
    transform = build_transform(mean=mean, std=std)
    train_and_valid_data = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    split_generator = torch.Generator().manual_seed(seed)
    train_data, valid_data = random_split(
        train_and_valid_data,
        [train_size, valid_size],
        generator=split_generator,
    )
    return train_data, valid_data, test_data


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_size: int,
    valid_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_data, valid_data, test_data = build_datasets(
        data_root=data_root,
        seed=seed,
        train_size=train_size,
        valid_size=valid_size,
        mean=mean,
        std=std,
    )

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, valid_loader, test_loader
