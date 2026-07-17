from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def build_transform(image_size: int) -> Compose:
    return Compose(
        [
            Resize(image_size),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def build_dataset(data_root: Path, image_size: int) -> CIFAR10:
    return CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=build_transform(image_size),
    )


def build_dataloader(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    image_size: int,
) -> DataLoader:
    train_dataset = build_dataset(data_root, image_size)
    loader_generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

