import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def convert_trimap_to_classes(mask: object) -> torch.Tensor:
    mask_array = np.array(mask, dtype=np.int64)
    class_mask = np.zeros_like(mask_array, dtype=np.int64)
    class_mask[mask_array == 1] = 1
    class_mask[mask_array == 2] = 0
    class_mask[mask_array == 3] = 2
    return torch.from_numpy(class_mask).long()


class PetSegmentationTransform:
    def __init__(
        self,
        image_size: int,
        use_augmentation: bool,
    ) -> None:
        self.image_size = image_size
        self.use_augmentation = use_augmentation

    def __call__(
        self,
        image: object,
        mask: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = F.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = F.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        if self.use_augmentation and random.random() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = convert_trimap_to_classes(mask)
        return image_tensor, mask_tensor


class PetSegmentationDataset(Dataset):
    def __init__(
        self,
        base_dataset: OxfordIIITPet,
        transform: PetSegmentationTransform,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.base_dataset[index]
        return self.transform(image, mask)


def build_datasets(
    data_root: Path,
    valid_ratio: float,
    seed: int,
    image_size: int,
    use_augmentation: bool,
) -> tuple[Subset, Subset, PetSegmentationDataset]:
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1.")

    trainval_base = OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types="segmentation",
        download=True,
    )
    train_dataset_full = PetSegmentationDataset(
        trainval_base,
        PetSegmentationTransform(image_size, use_augmentation),
    )
    valid_dataset_full = PetSegmentationDataset(
        trainval_base,
        PetSegmentationTransform(image_size, use_augmentation=False),
    )
    test_dataset = PetSegmentationDataset(
        OxfordIIITPet(
            root=data_root,
            split="test",
            target_types="segmentation",
            download=True,
        ),
        PetSegmentationTransform(image_size, use_augmentation=False),
    )

    valid_size = int(len(trainval_base) * valid_ratio)
    if valid_size == 0 or valid_size >= len(trainval_base):
        raise ValueError("valid_ratio produces an invalid validation size.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(trainval_base), generator=generator).tolist()
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
    image_size: int,
    use_augmentation: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, valid_dataset, test_dataset = build_datasets(
        data_root=data_root,
        valid_ratio=valid_ratio,
        seed=seed,
        image_size=image_size,
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
