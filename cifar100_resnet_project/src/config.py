from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Training settings shared by the project scripts."""

    data_root: str = "datasets"
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
