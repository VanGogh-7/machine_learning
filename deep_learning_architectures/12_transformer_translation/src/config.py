from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "multi30k"
    train_src_path: Path = data_root / "train.en"
    train_tgt_path: Path = data_root / "train.de"
    valid_src_path: Path = data_root / "valid.en"
    valid_tgt_path: Path = data_root / "valid.de"
    test_src_path: Path = data_root / "test.en"
    test_tgt_path: Path = data_root / "test.de"
    batch_size: int = 64
    seed: int = 42
    src_vocab_min_freq: int = 2
    tgt_vocab_min_freq: int = 2
    max_src_length: int = 64
    max_tgt_length: int = 64
    embedding_dim: int = 256
    num_heads: int = 8
    feedforward_dim: int = 512
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    n_epochs: int = 10
    clip_grad_norm: float = 1.0
    model_path: str = "transformer_translation.pt"
    src_vocab_path: str = "src_vocab.json"
    tgt_vocab_path: str = "tgt_vocab.json"
    history_path: str = "training_history.json"
