from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "cifar10"
    batch_size: int = 64
    num_workers: int = 2
    seed: int = 42
    in_channels: int = 3
    n_classes: int = 10
    learning_rate: float = 1e-3
    n_epochs: int = 10
    model_path: str = "ch3.pt"
