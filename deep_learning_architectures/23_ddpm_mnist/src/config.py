from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "mnist"
    batch_size: int = 128
    seed: int = 42
    valid_ratio: float = 0.1
    num_workers: int = 2
    image_size: int = 28
    image_channels: int = 1
    base_channels: int = 64
    time_embedding_dim: int = 128
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    n_epochs: int = 20
    model_path: str = "ddpm_mnist.pt"
    history_path: str = "training_history.json"
    output_dir: str = "outputs"
    sample_interval: int = 5
    num_generation_samples: int = 64

    def __post_init__(self) -> None:
        if not 0 < self.valid_ratio < 1:
            raise ValueError("valid_ratio must be between 0 and 1.")
        if self.num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive.")
        if not 0 < self.beta_start < self.beta_end < 1:
            raise ValueError("beta_start and beta_end must satisfy 0 < start < end < 1.")
        if self.sample_interval <= 0:
            raise ValueError("sample_interval must be positive.")

