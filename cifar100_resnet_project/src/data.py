from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def build_train_transform() -> v2.Compose:
    """Build the augmentation and normalization pipeline for training."""
    return v2.Compose(
        [
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def build_eval_transform() -> v2.Compose:
    """Build the deterministic normalization pipeline for evaluation."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def build_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    valid_fraction: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Download CIFAR-100 and create reproducible train, validation, and test loaders."""
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1.")

    root = Path(data_root)
    train_data = CIFAR100(root=root, train=True, download=True, transform=build_train_transform())
    valid_data = CIFAR100(root=root, train=True, download=True, transform=build_eval_transform())
    test_data = CIFAR100(root=root, train=False, download=True, transform=build_eval_transform())

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_data), generator=generator).tolist()
    valid_size = int(len(indices) * valid_fraction)
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]

    train_subset = Subset(train_data, train_indices)
    valid_subset = Subset(valid_data, valid_indices)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_subset, shuffle=True, generator=generator, **loader_options)
    valid_loader = DataLoader(valid_subset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)
    return train_loader, valid_loader, test_loader
