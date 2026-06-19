from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import TrainConfig
from src.utils import load_json


REQUIRED_PROCESSED_COLUMNS = {"user_id", "item_id"}


class ProcessedTwoTowerDataset(Dataset):
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.dataframe = pd.read_csv(
            csv_path,
            dtype={
                "user_id": "int64",
                "item_id": "int64",
            },
        )

        missing = REQUIRED_PROCESSED_COLUMNS - set(self.dataframe.columns)
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing)}"
            )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        return (
            torch.tensor(int(row["user_id"]), dtype=torch.long),
            torch.tensor(int(row["item_id"]), dtype=torch.long),
        )


def processed_files_exist(config: TrainConfig) -> bool:
    return all(
        path.is_file()
        for path in (
            config.processed_train_path,
            config.processed_valid_path,
            config.processed_test_path,
            config.processed_feature_meta_path,
        )
    )


def require_processed_files(config: TrainConfig) -> None:
    if processed_files_exist(config):
        return

    missing_paths = [
        path
        for path in (
            config.processed_train_path,
            config.processed_valid_path,
            config.processed_test_path,
            config.processed_feature_meta_path,
        )
        if not path.is_file()
    ]
    missing_text = "\n".join(f"  - {path}" for path in missing_paths)
    raise FileNotFoundError(
        "Processed Two-Tower CSV files were not found.\n"
        f"Missing files:\n{missing_text}\n"
        "Please run:\n"
        "python scripts/prepare_data.py"
    )


def build_dataloaders(
    config: TrainConfig,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    require_processed_files(config)
    metadata = load_json(config.processed_feature_meta_path)

    train_dataset = ProcessedTwoTowerDataset(config.processed_train_path)
    valid_dataset = ProcessedTwoTowerDataset(config.processed_valid_path)
    test_dataset = ProcessedTwoTowerDataset(config.processed_test_path)

    pin_memory = device is not None and device.type == "cuda"
    loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, valid_loader, test_loader, metadata
