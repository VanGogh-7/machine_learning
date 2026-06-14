from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "mnist"
    checkpoint_name: str = "lenet5_mnist.pt"
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 42
    n_classes: int = 10
    learning_rate: float = 0.001
    n_epochs: int = 10
    train_size: int = 55_000
    valid_size: int = 5_000
    mean: tuple[float, ...] = (0.1307,)
    std: tuple[float, ...] = (0.3081,)
