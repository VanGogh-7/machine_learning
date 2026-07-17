from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
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


def build_datasets(
    data_root: Path,
    valid_size: int,
    seed: int,
) -> tuple[Subset, Subset, CIFAR10]:
    train_dataset_full = CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=build_train_transform(),
    )
    valid_dataset_full = CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=build_test_transform(),
    )
    test_dataset = CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=build_test_transform(),
    )

    if not 0 < valid_size < len(train_dataset_full):
        raise ValueError("valid_size must be between 1 and the training dataset size.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset_full), generator=generator).tolist()
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]

    train_dataset = Subset(train_dataset_full, train_indices)
    valid_dataset = Subset(valid_dataset_full, valid_indices)
    return train_dataset, valid_dataset, test_dataset


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    valid_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, valid_dataset, test_dataset = build_datasets(
        data_root=data_root,
        valid_size=valid_size,
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
