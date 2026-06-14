from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_root: str = "datasets"
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42
    n_inputs: int = 1 * 28 * 28
    n_hidden1: int = 188
    n_hidden2: int = 188
    n_classes: int = 10
    learning_rate: float = 0.0084
    n_epochs: int = 50
