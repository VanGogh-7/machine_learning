from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    """Training settings shared by the project scripts."""

    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "cifar100"
    batch_size: int = 128
    num_workers: int = 2
    seed: int = 42
    in_channels: int = 3
    n_classes: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 20
    model_path: str = "checkpoints/best_model.pt"
    image_size: int = 32
