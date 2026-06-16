from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "cifar10"
    batch_size: int = 128
    seed: int = 42
    num_workers: int = 2
    image_size: int = 32
    image_channels: int = 3
    latent_dim: int = 100
    generator_feature_maps: int = 64
    discriminator_feature_maps: int = 64
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    n_epochs: int = 50
    generator_path: str = "generator_dcgan_cifar10.pt"
    discriminator_path: str = "discriminator_dcgan_cifar10.pt"
    history_path: str = "training_history.json"
    output_dir: str = "outputs"
    sample_interval: int = 5
    num_generation_samples: int = 64

    def __post_init__(self) -> None:
        if self.image_size != 32:
            raise ValueError("This educational DCGAN architecture expects 32x32 images.")
        if self.sample_interval <= 0:
            raise ValueError("sample_interval must be positive.")

