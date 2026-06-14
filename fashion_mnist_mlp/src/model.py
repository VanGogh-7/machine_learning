import torch
from torch import nn


class ImageClassifier(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_hidden1: int,
        n_hidden2: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)
