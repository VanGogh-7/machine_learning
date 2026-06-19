from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets"
    raw_data_root: Path = data_root / "ml-10M100K"
    ratings_path: Path = raw_data_root / "ratings.dat"
    movies_path: Path = raw_data_root / "movies.dat"
    processed_data_root: Path = raw_data_root / "processed_lightgcn"
    processed_train_path: Path = processed_data_root / "train.csv"
    processed_valid_path: Path = processed_data_root / "valid.csv"
    processed_test_path: Path = processed_data_root / "test.csv"
    processed_feature_meta_path: Path = processed_data_root / "feature_metadata.json"
    graph_cache_path: Path = processed_data_root / "normalized_adj.pt"

    # Safe defaults keep the first MovieLens 10M run laptop-friendly.
    batch_size: int = 4096
    num_workers: int = 0
    seed: int = 42
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    debug_mode: bool = True
    max_users: int | None = 5000
    max_interactions: int | None = 500000
    max_edges: int | None = 300000
    save_large_metadata: bool = False

    positive_rating_threshold: float = 4.0
    negative_sampling_max_trials: int = 50

    embedding_dim: int = 64
    num_layers: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    bpr_reg_weight: float = 1e-4
    n_epochs: int = 3

    model_path: str = "lightgcn.pt"
    feature_meta_path: str = "feature_metadata.json"
    history_path: str = "training_history.json"
    output_dir: str = "outputs"
    top_k: int = 10
    eval_max_users: int = 1000
