from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int, use_augmentation: bool) -> Compose:
    if use_augmentation:
        return Compose(
            [
                Resize(image_size + 32),
                RandomResizedCrop(image_size),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return build_eval_transform(image_size)


def build_eval_transform(image_size: int) -> Compose:
    return Compose(
        [
            Resize(image_size + 32),
            CenterCrop(image_size),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_datasets(
    data_root: Path,
    image_size: int,
    use_augmentation: bool,
) -> tuple[Flowers102, Flowers102, Flowers102]:
    train_dataset = Flowers102(
        root=data_root,
        split="train",
        download=True,
        transform=build_train_transform(image_size, use_augmentation),
    )
    valid_dataset = Flowers102(
        root=data_root,
        split="val",
        download=True,
        transform=build_eval_transform(image_size),
    )
    test_dataset = Flowers102(
        root=data_root,
        split="test",
        download=True,
        transform=build_eval_transform(image_size),
    )
    return train_dataset, valid_dataset, test_dataset


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    image_size: int,
    use_augmentation: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, valid_dataset, test_dataset = build_datasets(
        data_root=data_root,
        image_size=image_size,
        use_augmentation=use_augmentation,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return train_loader, valid_loader, test_loader
