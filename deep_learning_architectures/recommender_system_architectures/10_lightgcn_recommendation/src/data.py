import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import TrainConfig
from src.utils import load_json


REQUIRED_PROCESSED_COLUMNS = {"user_id", "item_id"}


class LightGCNDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        num_items: int,
        negative_sampling_max_trials: int,
    ) -> None:
        self.csv_path = csv_path
        self.num_items = num_items
        self.negative_sampling_max_trials = negative_sampling_max_trials
        self.dataframe = pd.read_csv(
            csv_path,
            dtype={"user_id": "int64", "item_id": "int64"},
        )

        missing = REQUIRED_PROCESSED_COLUMNS - set(self.dataframe.columns)
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing)}"
            )

        self.edges = list(
            zip(
                self.dataframe["user_id"].astype(int).tolist(),
                self.dataframe["item_id"].astype(int).tolist(),
            )
        )
        self.user_positive_items = build_user_positive_items(self.dataframe)

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_id, positive_item_id = self.edges[index]
        negative_item_id = self.sample_negative_item(user_id)
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(positive_item_id, dtype=torch.long),
            torch.tensor(negative_item_id, dtype=torch.long),
        )

    def sample_negative_item(self, user_id: int) -> int:
        user_items = self.user_positive_items.get(user_id, set())
        for _ in range(self.negative_sampling_max_trials):
            candidate = random.randrange(self.num_items)
            if candidate not in user_items:
                return candidate

        # Rare fallback when rejection sampling misses repeatedly.
        for candidate in range(self.num_items):
            if candidate not in user_items:
                return candidate
        return 0


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
        "Processed LightGCN files were not found.\n"
        f"Missing files:\n{missing_text}\n"
        "Please run:\n"
        "python scripts/prepare_data.py"
    )


def read_positive_edges(csv_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        csv_path,
        dtype={"user_id": "int64", "item_id": "int64"},
    )
    missing = REQUIRED_PROCESSED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    return dataframe


def build_user_positive_items(edges: pd.DataFrame) -> dict[int, set[int]]:
    user_positive_items: dict[int, set[int]] = {}
    for user_id, user_edges in edges.groupby("user_id", sort=False):
        user_positive_items[int(user_id)] = set(
            user_edges["item_id"].astype(int).tolist()
        )
    return user_positive_items


def build_normalized_adj(
    train_edges: pd.DataFrame,
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    # The graph has user nodes [0, num_users) and item nodes offset by num_users.
    user_nodes = torch.tensor(train_edges["user_id"].to_numpy(), dtype=torch.long)
    item_nodes = torch.tensor(train_edges["item_id"].to_numpy(), dtype=torch.long)
    item_nodes = item_nodes + num_users
    row_indices = torch.cat([user_nodes, item_nodes])
    col_indices = torch.cat([item_nodes, user_nodes])

    num_nodes = num_users + num_items
    degrees = torch.bincount(row_indices, minlength=num_nodes).float()
    degree_inv_sqrt = degrees.clamp_min(1.0).pow(-0.5)
    values = degree_inv_sqrt[row_indices] * degree_inv_sqrt[col_indices]

    indices = torch.stack([row_indices, col_indices])
    normalized_adj = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(num_nodes, num_nodes),
    )
    return normalized_adj.coalesce()


def build_lightgcn_data(
    config: TrainConfig,
    device: torch.device | None = None,
) -> tuple[
    DataLoader,
    pd.DataFrame,
    pd.DataFrame,
    torch.Tensor,
    dict[str, Any],
    dict[int, set[int]],
]:
    require_processed_files(config)
    metadata = load_json(config.processed_feature_meta_path)

    train_edges = read_positive_edges(config.processed_train_path)
    valid_edges = read_positive_edges(config.processed_valid_path)
    test_edges = read_positive_edges(config.processed_test_path)

    train_dataset = LightGCNDataset(
        csv_path=config.processed_train_path,
        num_items=int(metadata["num_items"]),
        negative_sampling_max_trials=config.negative_sampling_max_trials,
    )
    normalized_adj = build_normalized_adj(
        train_edges=train_edges,
        num_users=int(metadata["num_users"]),
        num_items=int(metadata["num_items"]),
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
    return (
        train_loader,
        valid_edges,
        test_edges,
        normalized_adj,
        metadata,
        train_dataset.user_positive_items,
    )
