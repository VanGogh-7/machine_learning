from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_root: str = "datasets"
    batch_size: int = 64
    num_workers: int = 2
    seed: int = 42
    in_channels: int = 1
    n_classes: int = 10
    learning_rate: float = 1e-3
    n_epochs: int = 5
    model_path: str = "mnist_cnn.pt"
