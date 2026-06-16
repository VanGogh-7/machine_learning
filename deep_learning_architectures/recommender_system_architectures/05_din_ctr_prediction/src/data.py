from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import TrainConfig
from src.utils import load_json


REQUIRED_PROCESSED_COLUMNS = {
    "target_item_id",
    "history_item_ids",
    "history_mask",
    "label",
}


class ProcessedDINDataset(Dataset):
    def __init__(self, csv_path: Path, max_history_length: int) -> None:
        self.csv_path = csv_path
        self.max_history_length = max_history_length
        self.dataframe = pd.read_csv(
            csv_path,
            dtype={
                "target_item_id": "int64",
                "history_item_ids": "string",
                "history_mask": "string",
                "label": "float32",
            },
        )

        missing = REQUIRED_PROCESSED_COLUMNS - set(self.dataframe.columns)
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing)}"
            )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        history_item_ids = parse_sequence(
            row["history_item_ids"],
            self.max_history_length,
            dtype=torch.long,
            column_name="history_item_ids",
        )
        history_mask = parse_sequence(
            row["history_mask"],
            self.max_history_length,
            dtype=torch.float32,
            column_name="history_mask",
        )
        return (
            torch.tensor(int(row["target_item_id"]), dtype=torch.long),
            history_item_ids,
            history_mask,
            torch.tensor(float(row["label"]), dtype=torch.float32),
        )


def parse_sequence(
    value: Any,
    expected_length: int,
    dtype: torch.dtype,
    column_name: str,
) -> torch.Tensor:
    values = str(value).split()
    if len(values) != expected_length:
        raise ValueError(
            f"{column_name} must contain {expected_length} values, "
            f"but found {len(values)}."
        )
    if dtype == torch.long:
        return torch.tensor([int(item) for item in values], dtype=dtype)
    return torch.tensor([float(item) for item in values], dtype=dtype)


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
        "Processed DIN CSV files were not found.\n"
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
    max_history_length = int(metadata["max_history_length"])

    train_dataset = ProcessedDINDataset(
        config.processed_train_path,
        max_history_length=max_history_length,
    )
    valid_dataset = ProcessedDINDataset(
        config.processed_valid_path,
        max_history_length=max_history_length,
    )
    test_dataset = ProcessedDINDataset(
        config.processed_test_path,
        max_history_length=max_history_length,
    )

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
