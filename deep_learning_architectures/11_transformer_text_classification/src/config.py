from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "text_classification"
    dataset_path: Path = data_root / "simple_sentiment.csv"
    batch_size: int = 32
    seed: int = 42
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    vocab_min_freq: int = 1
    max_length: int = 64
    embedding_dim: int = 128
    num_heads: int = 4
    feedforward_dim: int = 256
    num_encoder_layers: int = 2
    dropout: float = 0.1
    n_classes: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 10
    model_path: str = "transformer_text_classification.pt"
    vocab_path: str = "transformer_vocab.json"
