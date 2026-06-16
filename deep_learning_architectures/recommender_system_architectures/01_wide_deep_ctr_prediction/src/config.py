from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "criteo_ctr"
    train_path: Path = data_root / "train.csv"
    valid_path: Path = data_root / "valid.csv"
    test_path: Path = data_root / "test.csv"
    batch_size: int = 1024
    seed: int = 42
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    num_numerical_features: int = 13
    numerical_feature_names: tuple[str, ...] = tuple(
        f"I{index}" for index in range(1, 14)
    )
    categorical_feature_names: tuple[str, ...] = tuple(
        f"C{index}" for index in range(1, 27)
    )
    label_col: str = "label"
    min_category_freq: int = 2
    embedding_dim: int = 16
    deep_hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 10
    model_path: str = "wide_deep_ctr.pt"
    feature_meta_path: str = "feature_metadata.json"
    history_path: str = "training_history.json"
    output_dir: str = "outputs"
