import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig


CSV_COLUMNS = ["user_id", "item_id"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, sort_keys=True)


def validate_config(config: TrainConfig) -> None:
    if config.max_users is not None and config.max_users <= 0:
        raise ValueError("max_users must be positive when provided.")
    if config.max_interactions is not None and config.max_interactions <= 0:
        raise ValueError("max_interactions must be positive when provided.")
    if config.max_edges is not None and config.max_edges <= 0:
        raise ValueError("max_edges must be positive when provided.")
    if config.negative_sampling_max_trials <= 0:
        raise ValueError("negative_sampling_max_trials must be positive.")
    if config.embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")
    if config.num_layers < 0:
        raise ValueError("num_layers must be non-negative.")
    if config.top_k <= 0:
        raise ValueError("top_k must be positive.")
    if config.eval_max_users <= 0:
        raise ValueError("eval_max_users must be positive.")
    if (
        config.valid_ratio <= 0
        or config.test_ratio <= 0
        or config.valid_ratio + config.test_ratio >= 1
    ):
        raise ValueError(
            "valid_ratio and test_ratio must be positive and sum to less than 1."
        )


def read_raw_movielens(config: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.ratings_path.is_file() or not config.movies_path.is_file():
        raise FileNotFoundError(
            "Raw MovieLens 10M files were not found.\n"
            f"Expected ratings file: {config.ratings_path}\n"
            f"Expected movies file: {config.movies_path}"
        )

    ratings = pd.read_csv(
        config.ratings_path,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        config.movies_path,
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1",
    )

    ratings = ratings.astype(
        {
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int64",
        }
    )
    movies["movieId"] = movies["movieId"].astype("int32")
    return ratings, movies


def apply_safety_limits(
    ratings: pd.DataFrame,
    config: TrainConfig,
) -> pd.DataFrame:
    limited = ratings

    if config.max_users is not None:
        selected_user_ids = np.sort(limited["userId"].unique())[: config.max_users]
        limited = limited[limited["userId"].isin(selected_user_ids)]

    limited = limited.sort_values(["userId", "timestamp"])
    if config.max_interactions is not None and len(limited) > config.max_interactions:
        limited = limited.head(config.max_interactions)

    return limited.reset_index(drop=True)


def encode_edges(
    positive_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    config: TrainConfig,
) -> tuple[list[tuple[int, int]], dict[int, int], dict[int, int], dict[int, str]]:
    if positive_ratings.empty:
        raise ValueError("No positive interactions remain after preprocessing.")

    raw_user_ids = np.sort(positive_ratings["userId"].unique())
    raw_movie_ids = np.sort(positive_ratings["movieId"].unique())
    raw_user_id_to_user_id = {
        int(raw_user_id): index for index, raw_user_id in enumerate(raw_user_ids)
    }
    raw_movie_id_to_item_id = {
        int(raw_movie_id): index for index, raw_movie_id in enumerate(raw_movie_ids)
    }

    item_id_to_title: dict[int, str] = {}
    used_movie_ids = set(raw_movie_id_to_item_id)
    for row in movies[movies["movieId"].isin(used_movie_ids)].itertuples(index=False):
        raw_movie_id = int(getattr(row, "movieId"))
        item_id_to_title[raw_movie_id_to_item_id[raw_movie_id]] = str(
            getattr(row, "title")
        )

    edges: list[tuple[int, int]] = []
    positive_ratings = positive_ratings.sort_values(["userId", "timestamp"])
    for row in positive_ratings.itertuples(index=False):
        user_id = raw_user_id_to_user_id[int(getattr(row, "userId"))]
        item_id = raw_movie_id_to_item_id[int(getattr(row, "movieId"))]
        edges.append((user_id, item_id))
        if config.max_edges is not None and len(edges) >= config.max_edges:
            break

    return edges, raw_user_id_to_user_id, raw_movie_id_to_item_id, item_id_to_title


def split_edges(
    edges: list[tuple[int, int]],
    config: TrainConfig,
) -> tuple[list[tuple[int, int]], ...]:
    if len(edges) < 3:
        raise ValueError("At least three positive edges are required for splitting.")

    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(edges))
    shuffled = [edges[int(index)] for index in indices]

    test_size = max(1, int(len(shuffled) * config.test_ratio))
    valid_size = max(1, int(len(shuffled) * config.valid_ratio))
    if valid_size + test_size >= len(shuffled):
        raise ValueError("The generated LightGCN dataset is too small for the split.")

    train_end = len(shuffled) - valid_size - test_size
    valid_end = len(shuffled) - test_size
    return (
        shuffled[:train_end],
        shuffled[train_end:valid_end],
        shuffled[valid_end:],
    )


def write_split(path: Path, rows: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def build_metadata(
    config: TrainConfig,
    raw_user_id_to_user_id: dict[int, int],
    raw_movie_id_to_item_id: dict[int, int],
    item_id_to_title: dict[int, str],
    train_size: int,
    valid_size: int,
    test_size: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "dataset_name": "MovieLens 10M",
        "raw_data_root": str(config.raw_data_root),
        "processed_data_root": str(config.processed_data_root),
        "num_users": len(raw_user_id_to_user_id),
        "num_items": len(raw_movie_id_to_item_id),
        "num_train_edges": train_size,
        "num_valid_edges": valid_size,
        "num_test_edges": test_size,
        "positive_rating_threshold": config.positive_rating_threshold,
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
        "debug_mode": config.debug_mode,
        "max_users": config.max_users,
        "max_interactions": config.max_interactions,
        "max_edges": config.max_edges,
        "task_type": "implicit_feedback_graph_collaborative_filtering",
    }

    if config.save_large_metadata:
        metadata["item_id_to_title"] = {
            str(item_id): title for item_id, title in sorted(item_id_to_title.items())
        }

    return metadata


def main() -> None:
    config = TrainConfig()
    validate_config(config)
    set_seed(config.seed)

    print(f"Raw dataset path: {config.raw_data_root}")
    print(f"Processed output path: {config.processed_data_root}")
    print(f"debug_mode: {config.debug_mode}")
    print(f"active max_users: {config.max_users}")
    print(f"active max_interactions: {config.max_interactions}")
    print(f"active max_edges: {config.max_edges}")
    if not config.debug_mode:
        print("Warning: full MovieLens 10M graph preprocessing may take time and memory.")

    ratings, movies = read_raw_movielens(config)
    print(f"Raw ratings loaded: {len(ratings)}")

    ratings = apply_safety_limits(ratings, config)
    if config.debug_mode:
        print("Debug limits are active.")
    elif any(
        value is not None
        for value in (config.max_users, config.max_interactions, config.max_edges)
    ):
        print("Explicit safety limits are active even with debug_mode=False.")

    print(f"Ratings after filtering: {len(ratings)}")
    positive_ratings = ratings[
        ratings["rating"] >= config.positive_rating_threshold
    ].reset_index(drop=True)
    print(f"Positive interactions: {len(positive_ratings)}")
    print(f"Users used: {positive_ratings['userId'].nunique()}")
    print(f"Unique movies used: {positive_ratings['movieId'].nunique()}")

    edges, raw_user_id_to_user_id, raw_movie_id_to_item_id, item_id_to_title = (
        encode_edges(
            positive_ratings=positive_ratings,
            movies=movies,
            config=config,
        )
    )
    train_rows, valid_rows, test_rows = split_edges(edges, config)
    write_split(config.processed_train_path, train_rows)
    write_split(config.processed_valid_path, valid_rows)
    write_split(config.processed_test_path, test_rows)

    metadata = build_metadata(
        config=config,
        raw_user_id_to_user_id=raw_user_id_to_user_id,
        raw_movie_id_to_item_id=raw_movie_id_to_item_id,
        item_id_to_title=item_id_to_title,
        train_size=len(train_rows),
        valid_size=len(valid_rows),
        test_size=len(test_rows),
    )
    save_json(metadata, config.processed_feature_meta_path)

    print(f"Generated positive graph edges: {len(edges)}")
    print(f"Train size: {len(train_rows)}")
    print(f"Validation size: {len(valid_rows)}")
    print(f"Test size: {len(test_rows)}")
    print(f"Saved train CSV: {config.processed_train_path}")
    print(f"Saved validation CSV: {config.processed_valid_path}")
    print(f"Saved test CSV: {config.processed_test_path}")
    print(f"Saved feature metadata: {config.processed_feature_meta_path}")


if __name__ == "__main__":
    main()
