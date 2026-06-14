import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.engine import train
from src.model import SmallResNet
from src.utils import clear_memory, get_device, set_seed
from src.visualize import plot_learning_curves


def main() -> None:
    config = TrainConfig()
    clear_memory()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, valid_loader, _ = build_dataloaders(
        config.data_root, config.batch_size, config.num_workers, config.seed
    )
    model = SmallResNet(config.in_channels, config.n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs)

    history = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        n_epochs=config.n_epochs,
        model_path=config.model_path,
    )
    plot_learning_curves(history)


if __name__ == "__main__":
    main()
