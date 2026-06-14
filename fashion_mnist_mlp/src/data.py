from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2


def build_transform() -> Callable:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = build_transform()
    train_and_valid_data = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.FashionMNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    split_generator = torch.Generator().manual_seed(seed)
    train_data, valid_data = random_split(
        train_and_valid_data,
        [55_000, 5_000],
        generator=split_generator,
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
