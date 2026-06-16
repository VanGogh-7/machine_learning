from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_train_transform() -> Compose:
    return Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_test_transform() -> Compose:
    return Compose(
        [
            ToTensor(),
            Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_datasets(data_root: Path) -> tuple[CIFAR10, CIFAR10]:
    train_data = CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=build_train_transform(),
    )
    test_data = CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=build_test_transform(),
    )
    return train_data, test_data


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_data, test_data = build_datasets(data_root)
    loader_generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=loader_generator,
        **loader_options,
    )
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)
    return train_loader, test_loader
