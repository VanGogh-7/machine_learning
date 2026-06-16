from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "oxford_iiit_pet"
    batch_size: int = 16
    seed: int = 42
    valid_ratio: float = 0.1
    num_workers: int = 2
    image_size: int = 128
    n_classes: int = 3
    base_channels: int = 32
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 20
    model_path: str = "unet_pet_segmentation.pt"
    history_path: str = "training_history.json"
    output_dir: str = "outputs"
    use_augmentation: bool = True
