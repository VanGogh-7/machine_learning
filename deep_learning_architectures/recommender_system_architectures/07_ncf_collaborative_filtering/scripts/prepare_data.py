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


CSV_COLUMNS = ["user_id", "item_id", "label"]


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
    if config.max_samples is not None and config.max_samples <= 0:
        raise ValueError("max_samples must be positive when provided.")
    if config.negative_samples_per_positive < 0:
        raise ValueError("negative_samples_per_positive must be non-negative.")
    if config.negative_sampling_max_trials <= 0:
        raise ValueError("negative_sampling_max_trials must be positive.")
    if (
        config.valid_ratio <= 0
        or config.test_ratio <= 0
        or config.valid_ratio + config.test_ratio >= 1
    ):
        raise ValueError(
            "valid_ratio and test_ratio must be positive and sum to less than 1."
        )
    if config.user_embedding_dim != config.item_embedding_dim:
        raise ValueError(
            "user_embedding_dim and item_embedding_dim must match for the GMF branch."
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


def encode_ids(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, int], dict[int, int], dict[int, str]]:
    if ratings.empty:
        raise ValueError("No ratings remain after applying preprocessing limits.")

    raw_user_ids = np.sort(ratings["userId"].unique())
    raw_movie_ids = np.sort(ratings["movieId"].unique())
    raw_user_id_to_user_id = {
        int(raw_user_id): index for index, raw_user_id in enumerate(raw_user_ids)
    }
    raw_movie_id_to_item_id = {
        int(raw_movie_id): index for index, raw_movie_id in enumerate(raw_movie_ids)
    }

    encoded = ratings.copy()
    encoded["user_id"] = (
        encoded["userId"].map(raw_user_id_to_user_id).astype("int32")
    )
    encoded["item_id"] = (
        encoded["movieId"].map(raw_movie_id_to_item_id).astype("int32")
    )

    item_id_to_title: dict[int, str] = {}
    used_movie_ids = set(raw_movie_id_to_item_id)
    for row in movies[movies["movieId"].isin(used_movie_ids)].itertuples(index=False):
        raw_movie_id = int(getattr(row, "movieId"))
        item_id_to_title[raw_movie_id_to_item_id[raw_movie_id]] = str(
            getattr(row, "title")
        )

    return encoded, raw_user_id_to_user_id, raw_movie_id_to_item_id, item_id_to_title


def sample_negative_item(
    num_items: int,
    user_interacted_items: set[int],
    rng: np.random.Generator,
    max_trials: int,
) -> int | None:
    for _ in range(max_trials):
        candidate = int(rng.integers(0, num_items))
        if candidate not in user_interacted_items:
            return candidate
    return None


def build_user_interactions(ratings: pd.DataFrame) -> dict[int, set[int]]:
    user_interacted_items: dict[int, set[int]] = {}
    for user_id, user_history in ratings.groupby("user_id", sort=False):
        user_interacted_items[int(user_id)] = set(
            user_history["item_id"].astype(int).tolist()
        )
    return user_interacted_items


def build_samples(
    ratings: pd.DataFrame,
    config: TrainConfig,
) -> tuple[list[tuple[int, int, int]], int]:
    rng = np.random.default_rng(config.seed)
    num_items = int(ratings["item_id"].nunique())
    user_interacted_items = build_user_interactions(ratings)
    samples: list[tuple[int, int, int]] = []
    positive_samples = 0

    positive_ratings = ratings[ratings["rating"] >= config.positive_rating_threshold]
    positive_ratings = positive_ratings.sort_values(["user_id", "timestamp"])

    for row in positive_ratings.itertuples(index=False):
        user_id = int(getattr(row, "user_id"))
        item_id = int(getattr(row, "item_id"))
        samples.append((user_id, item_id, 1))
        positive_samples += 1
        if config.max_samples is not None and len(samples) >= config.max_samples:
            return samples[: config.max_samples], positive_samples

        for _ in range(config.negative_samples_per_positive):
            negative_item_id = sample_negative_item(
                num_items=num_items,
                user_interacted_items=user_interacted_items[user_id],
                rng=rng,
                max_trials=config.negative_sampling_max_trials,
            )
            if negative_item_id is None:
                continue
            samples.append((user_id, negative_item_id, 0))
            if config.max_samples is not None and len(samples) >= config.max_samples:
                return samples[: config.max_samples], positive_samples

    return samples, positive_samples


def split_samples(
    samples: list[tuple[int, int, int]],
    config: TrainConfig,
) -> tuple[list[tuple[int, int, int]], ...]:
    if len(samples) < 3:
        raise ValueError("At least three samples are required for train/valid/test split.")

    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(samples))
    shuffled = [samples[int(index)] for index in indices]

    test_size = max(1, int(len(shuffled) * config.test_ratio))
    valid_size = max(1, int(len(shuffled) * config.valid_ratio))
    if valid_size + test_size >= len(shuffled):
        raise ValueError("The generated NCF dataset is too small for the split.")

    train_end = len(shuffled) - valid_size - test_size
    valid_end = len(shuffled) - test_size
    return (
        shuffled[:train_end],
        shuffled[train_end:valid_end],
        shuffled[valid_end:],
    )


def write_split(path: Path, rows: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def build_metadata(
    config: TrainConfig,
    ratings: pd.DataFrame,
    item_id_to_title: dict[int, str],
    train_size: int,
    valid_size: int,
    test_size: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "dataset_name": "MovieLens 10M",
        "raw_data_root": str(config.raw_data_root),
        "processed_data_root": str(config.processed_data_root),
        "num_users": int(ratings["user_id"].nunique()),
        "num_items": int(ratings["item_id"].nunique()),
        "positive_rating_threshold": config.positive_rating_threshold,
        "negative_samples_per_positive": config.negative_samples_per_positive,
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
        "debug_mode": config.debug_mode,
        "max_users": config.max_users,
        "max_interactions": config.max_interactions,
        "max_samples": config.max_samples,
        "task_type": "implicit_feedback_binary_classification",
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
    print(f"active max_samples: {config.max_samples}")
    if not config.debug_mode:
        print("Warning: full MovieLens 10M preprocessing may take time and memory.")

    ratings, movies = read_raw_movielens(config)
    print(f"Raw ratings loaded: {len(ratings)}")

    ratings = apply_safety_limits(ratings, config)
    if config.debug_mode:
        print("Debug limits are active.")
    elif any(
        value is not None
        for value in (config.max_users, config.max_interactions, config.max_samples)
    ):
        print("Explicit safety limits are active even with debug_mode=False.")

    ratings = ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)
    print(f"Ratings after filtering: {len(ratings)}")
    print(f"Users used: {ratings['userId'].nunique()}")
    print(f"Unique movies used: {ratings['movieId'].nunique()}")

    ratings, _, _, item_id_to_title = encode_ids(ratings, movies)
    samples, positive_samples = build_samples(ratings, config)
    if not samples:
        raise ValueError(
            "No NCF samples were generated. Try lowering positive_rating_threshold."
        )

    train_rows, valid_rows, test_rows = split_samples(samples, config)
    write_split(config.processed_train_path, train_rows)
    write_split(config.processed_valid_path, valid_rows)
    write_split(config.processed_test_path, test_rows)

    metadata = build_metadata(
        config=config,
        ratings=ratings,
        item_id_to_title=item_id_to_title,
        train_size=len(train_rows),
        valid_size=len(valid_rows),
        test_size=len(test_rows),
    )
    save_json(metadata, config.processed_feature_meta_path)

    print(f"Positive samples: {positive_samples}")
    print(f"Generated samples: {len(samples)}")
    print(f"Train size: {len(train_rows)}")
    print(f"Validation size: {len(valid_rows)}")
    print(f"Test size: {len(test_rows)}")
    print(f"Saved train CSV: {config.processed_train_path}")
    print(f"Saved validation CSV: {config.processed_valid_path}")
    print(f"Saved test CSV: {config.processed_test_path}")
    print(f"Saved feature metadata: {config.processed_feature_meta_path}")


if __name__ == "__main__":
    main()
