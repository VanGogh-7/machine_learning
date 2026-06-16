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
    n_classes: int = 102
    width_multiplier: float = 1.0
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    n_epochs: int = 20
    model_path: str = "mobilenet_flowers102.pt"
    history_path: str = "training_history.json"
    output_dir: str = "outputs"
    use_augmentation: bool = True
