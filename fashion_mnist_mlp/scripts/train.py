import sys
from pathlib import Path

import torch
from torch import nn
from torchmetrics import Accuracy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.engine import train
from src.model import ImageClassifier
from src.utils import clear_memory, get_device, set_seed
from src.visualize import plot_learning_curves


def main() -> None:
    config = TrainConfig()
    clear_memory()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, valid_loader, _ = build_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    model = ImageClassifier(
        n_inputs=config.n_inputs,
        n_hidden1=config.n_hidden1,
        n_hidden2=config.n_hidden2,
        n_classes=config.n_classes,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    train_metric = Accuracy(task="multiclass", num_classes=config.n_classes).to(device)
    valid_metric = Accuracy(task="multiclass", num_classes=config.n_classes).to(device)

    history = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_metric=train_metric,
        valid_metric=valid_metric,
        device=device,
        n_epochs=config.n_epochs,
    )
    plot_learning_curves(history, config.n_epochs)
    torch.save(model.state_dict(), PROJECT_ROOT / "fashion_mnist_mlp.pt")
    print(f"Saved model to {PROJECT_ROOT / 'fashion_mnist_mlp.pt'}")


if __name__ == "__main__":
    main()
