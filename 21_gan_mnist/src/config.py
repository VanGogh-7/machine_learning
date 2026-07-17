from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "mnist"
    batch_size: int = 128
    seed: int = 42
    num_workers: int = 2
    image_size: int = 28
    image_dim: int = 784
    in_channels: int = 1
    latent_dim: int = 100
    generator_hidden_dims: tuple[int, ...] = (256, 512, 1024)
    discriminator_hidden_dims: tuple[int, ...] = (512, 256)
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    n_epochs: int = 50
    generator_path: str = "generator_mnist.pt"
    discriminator_path: str = "discriminator_mnist.pt"
    history_path: str = "training_history.json"
    output_dir: str = "outputs"
    sample_interval: int = 5
    num_generation_samples: int = 64

    def __post_init__(self) -> None:
        expected_dim = self.in_channels * self.image_size * self.image_size
        if self.image_dim != expected_dim:
            raise ValueError("image_dim must equal in_channels * image_size * image_size.")
        if self.sample_interval <= 0:
            raise ValueError("sample_interval must be positive.")

