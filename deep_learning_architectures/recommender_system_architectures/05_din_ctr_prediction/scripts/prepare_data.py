import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.utils import save_json, set_seed


PADDING_ITEM_ID = 0
CSV_COLUMNS = ["target_item_id", "history_item_ids", "history_mask", "label"]


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
    if config.max_history_length <= 0:
        raise ValueError("max_history_length must be positive.")
    if config.min_history_length <= 0:
        raise ValueError("min_history_length must be positive.")
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

    if config.max_interactions is not None and len(limited) > config.max_interactions:
        limited = limited.sort_values(["userId", "timestamp"]).head(
            config.max_interactions
        )

    return limited.reset_index(drop=True)


def encode_items(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, int], dict[int, str]]:
    if ratings.empty:
        raise ValueError("No ratings remain after applying preprocessing limits.")

    raw_movie_ids = np.sort(ratings["movieId"].unique())
    item_to_id = {
        int(raw_movie_id): index + 1
        for index, raw_movie_id in enumerate(raw_movie_ids)
    }

    encoded = ratings.copy()
    encoded["item_id"] = encoded["movieId"].map(item_to_id).astype("int32")

    title_lookup: dict[int, str] = {}
    for row in movies[movies["movieId"].isin(set(item_to_id))].itertuples(
        index=False
    ):
        raw_movie_id = int(getattr(row, "movieId"))
        title_lookup[item_to_id[raw_movie_id]] = str(getattr(row, "title"))

    return encoded, item_to_id, title_lookup


def format_history(
    positive_history: list[int],
    max_history_length: int,
) -> tuple[str, str]:
    truncated = positive_history[-max_history_length:]
    padding_length = max_history_length - len(truncated)
    padded_history = [PADDING_ITEM_ID] * padding_length + truncated
    history_mask = [0] * padding_length + [1] * len(truncated)
    return (
        " ".join(str(item_id) for item_id in padded_history),
        " ".join(str(mask_value) for mask_value in history_mask),
    )


def sample_negative_item(
    num_items: int,
    user_seen_items: set[int],
    rng: np.random.Generator,
    max_trials: int,
) -> int | None:
    for _ in range(max_trials):
        candidate = int(rng.integers(1, num_items + 1))
        if candidate != PADDING_ITEM_ID and candidate not in user_seen_items:
            return candidate
    return None


def build_samples(
    ratings: pd.DataFrame,
    config: TrainConfig,
) -> list[tuple[int, str, str, int]]:
    rng = np.random.default_rng(config.seed)
    num_items = int(ratings["item_id"].max())
    samples: list[tuple[int, str, str, int]] = []

    sorted_ratings = ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)
    for user_id, user_history in sorted_ratings.groupby("userId", sort=False):
        user_seen_items = set(user_history["item_id"].astype(int).tolist())
        positive_history: list[int] = []

        for row in user_history.itertuples(index=False):
            item_id = int(getattr(row, "item_id"))
            rating = float(getattr(row, "rating"))

            if rating < config.positive_rating_threshold:
                continue

            if len(positive_history) >= config.min_history_length:
                history_items, history_mask = format_history(
                    positive_history,
                    config.max_history_length,
                )
                samples.append((item_id, history_items, history_mask, 1))

                for _ in range(config.negative_samples_per_positive):
                    negative_item_id = sample_negative_item(
                        num_items=num_items,
                        user_seen_items=user_seen_items,
                        rng=rng,
                        max_trials=config.negative_sampling_max_trials,
                    )
                    if negative_item_id is not None:
                        samples.append(
                            (negative_item_id, history_items, history_mask, 0)
                        )

                if config.max_samples is not None and len(samples) >= config.max_samples:
                    return samples[: config.max_samples]

            positive_history.append(item_id)

    return samples


def split_samples(
    samples: list[tuple[int, str, str, int]],
    config: TrainConfig,
) -> tuple[list[tuple[int, str, str, int]], ...]:
    if len(samples) < 3:
        raise ValueError("At least three samples are required for train/valid/test split.")

    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(samples))
    shuffled = [samples[int(index)] for index in indices]

    test_size = max(1, int(len(shuffled) * config.test_ratio))
    valid_size = max(1, int(len(shuffled) * config.valid_ratio))
    if valid_size + test_size >= len(shuffled):
        raise ValueError("The generated DIN dataset is too small for the split.")

    train_end = len(shuffled) - valid_size - test_size
    valid_end = len(shuffled) - test_size
    return (
        shuffled[:train_end],
        shuffled[train_end:valid_end],
        shuffled[valid_end:],
    )


def write_split(path: Path, rows: list[tuple[int, str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def build_metadata(
    config: TrainConfig,
    ratings: pd.DataFrame,
    item_to_id: dict[int, int],
    title_lookup: dict[int, str],
    train_size: int,
    valid_size: int,
    test_size: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "dataset_name": "MovieLens 10M",
        "raw_data_root": str(config.raw_data_root),
        "ratings_path": str(config.ratings_path),
        "movies_path": str(config.movies_path),
        "processed_data_root": str(config.processed_data_root),
        "processed_train_path": str(config.processed_train_path),
        "processed_valid_path": str(config.processed_valid_path),
        "processed_test_path": str(config.processed_test_path),
        "num_users": int(ratings["userId"].nunique()),
        "num_items": len(item_to_id) + 1,
        "padding_item_id": PADDING_ITEM_ID,
        "max_history_length": config.max_history_length,
        "positive_rating_threshold": config.positive_rating_threshold,
        "negative_samples_per_positive": config.negative_samples_per_positive,
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
        "debug_mode": config.debug_mode,
        "max_users": config.max_users,
        "max_interactions": config.max_interactions,
        "max_samples": config.max_samples,
        "sequence_padding": "left",
        "sequence_column_format": "space-separated integers",
    }

    if config.save_large_metadata:
        # This optional metadata is useful for readable examples, but it is kept
        # out of the default file because full MovieLens mappings can be large.
        metadata["item_id_to_title"] = {
            str(item_id): title
            for item_id, title in sorted(title_lookup.items())
        }
        metadata["raw_movie_id_to_item_id"] = {
            str(raw_movie_id): item_id
            for raw_movie_id, item_id in sorted(item_to_id.items())
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
    raw_rating_count = len(ratings)
    print(f"Raw ratings loaded: {raw_rating_count}")

    if config.debug_mode:
        ratings = apply_safety_limits(ratings, config)
        print("Debug limits are active.")
    else:
        ratings = apply_safety_limits(ratings, config)
        if any(
            value is not None
            for value in (config.max_users, config.max_interactions, config.max_samples)
        ):
            print("Explicit safety limits are active even with debug_mode=False.")

    print(f"Ratings after filtering: {len(ratings)}")
    print(f"Users used: {ratings['userId'].nunique()}")
    print(f"Unique movies used: {ratings['movieId'].nunique()}")

    ratings, item_to_id, title_lookup = encode_items(ratings, movies)
    samples = build_samples(ratings, config)
    if not samples:
        raise ValueError(
            "No DIN samples were generated. Try lowering positive_rating_threshold "
            "or min_history_length."
        )

    train_rows, valid_rows, test_rows = split_samples(samples, config)
    write_split(config.processed_train_path, train_rows)
    write_split(config.processed_valid_path, valid_rows)
    write_split(config.processed_test_path, test_rows)

    metadata = build_metadata(
        config=config,
        ratings=ratings,
        item_to_id=item_to_id,
        title_lookup=title_lookup,
        train_size=len(train_rows),
        valid_size=len(valid_rows),
        test_size=len(test_rows),
    )
    save_json(metadata, config.processed_feature_meta_path)

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
