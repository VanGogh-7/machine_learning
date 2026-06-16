from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def build_train_transform(use_augmentation: bool) -> Compose:
    transforms = []
    if use_augmentation:
        transforms.extend(
            [
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
            ]
        )
    transforms.extend(
        [
            ToTensor(),
            Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return Compose(transforms)


def build_test_transform() -> Compose:
    return Compose(
        [
            ToTensor(),
            Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def build_datasets(
    data_root: Path,
    valid_ratio: float,
    seed: int,
    use_augmentation: bool,
) -> tuple[Subset, Subset, CIFAR100]:
    train_dataset_full = CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=build_train_transform(use_augmentation),
    )
    valid_dataset_full = CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=build_test_transform(),
    )
    test_dataset = CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=build_test_transform(),
    )

    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1.")

    valid_size = int(len(train_dataset_full) * valid_ratio)
    if valid_size == 0 or valid_size >= len(train_dataset_full):
        raise ValueError("valid_ratio produces an invalid validation size.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset_full), generator=generator).tolist()
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]
    return (
        Subset(train_dataset_full, train_indices),
        Subset(valid_dataset_full, valid_indices),
        test_dataset,
    )


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    valid_ratio: float,
    use_augmentation: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, valid_dataset, test_dataset = build_datasets(
        data_root=data_root,
        valid_ratio=valid_ratio,
        seed=seed,
        use_augmentation=use_augmentation,
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
