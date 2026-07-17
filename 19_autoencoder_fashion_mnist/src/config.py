from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "fashion_mnist"
    batch_size: int = 128
    seed: int = 42
    valid_ratio: float = 0.1
    num_workers: int = 2
    image_size: int = 28
    in_channels: int = 1
    latent_dim: int = 64
    base_channels: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 20
    model_path: str = "autoencoder_fashion_mnist.pt"
    history_path: str = "training_history.json"
    output_dir: str = "outputs"

    def __post_init__(self) -> None:
        if not 0 < self.valid_ratio < 1:
            raise ValueError("valid_ratio must be between 0 and 1.")
        if self.image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4.")

