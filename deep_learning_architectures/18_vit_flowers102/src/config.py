from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "flowers102"
    batch_size: int = 64
    seed: int = 42
    num_workers: int = 2
    image_size: int = 128
    patch_size: int = 16
    n_classes: int = 102
    embedding_dim: int = 192
    num_heads: int = 6
    mlp_dim: int = 384
    num_encoder_layers: int = 6
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    n_epochs: int = 20
    model_path: str = "vit_flowers102.pt"
    history_path: str = "training_history.json"
    output_dir: str = "outputs"
    use_augmentation: bool = True

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

