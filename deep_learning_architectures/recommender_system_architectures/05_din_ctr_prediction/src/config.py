from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class TrainConfig:
    # MovieLens data is stored in the repository-level datasets/ directory.
    datasets_root: Path = REPO_ROOT / "datasets"
    raw_data_root: Path = datasets_root / "ml-10M100K"
    ratings_path: Path = raw_data_root / "ratings.dat"
    movies_path: Path = raw_data_root / "movies.dat"
    processed_data_root: Path = raw_data_root / "processed_din"
    processed_train_path: Path = processed_data_root / "train.csv"
    processed_valid_path: Path = processed_data_root / "valid.csv"
    processed_test_path: Path = processed_data_root / "test.csv"
    processed_feature_meta_path: Path = processed_data_root / "feature_metadata.json"

    # MovieLens 10M is large. Defaults are intentionally debug-friendly so the
    # first run does not generate millions of samples on a laptop.
    debug_mode: bool = False
    max_users: int | None = 2000
    max_interactions: int | None = 300000
    max_samples: int | None = 100000
    save_large_metadata: bool = False

    batch_size: int = 512
    num_workers: int = 2
    seed: int = 42
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    positive_rating_threshold: float = 4.0
    negative_samples_per_positive: int = 1
    negative_sampling_max_trials: int = 50
    max_history_length: int = 20
    min_history_length: int = 1
    embedding_dim: int = 32
    activation_hidden_dims: tuple[int, ...] = (64, 32)
    mlp_hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 20
    model_path: str = "din_ctr.pt"
    feature_meta_path: str = "feature_metadata.json"
    history_path: str = "training_history.json"
    output_dir: str = "outputs"
