from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "multi30k"
    batch_size: int = 64
    seed: int = 42
    src_language: str = "en"
    tgt_language: str = "de"
    src_vocab_min_freq: int = 2
    tgt_vocab_min_freq: int = 2
    max_src_length: int = 40
    max_tgt_length: int = 40
    embedding_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.2
    teacher_forcing_ratio: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 10
    model_path: str = "seq2seq_attention_translation.pt"
