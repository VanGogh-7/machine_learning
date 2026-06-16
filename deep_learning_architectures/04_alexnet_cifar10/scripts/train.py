import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.engine import train
from src.model import AlexNetCIFAR10
from src.utils import clear_memory, get_device, set_seed


def main() -> None:
    config = TrainConfig()
    clear_memory()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    batch_inputs, batch_targets = next(iter(train_loader))
    print(f"Batch input shape: {tuple(batch_inputs.shape)}")
    print(f"Batch target shape: {tuple(batch_targets.shape)}")

    model = AlexNetCIFAR10(
        in_channels=config.in_channels,
        n_classes=config.n_classes,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_path = PROJECT_ROOT / config.model_path

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        n_epochs=config.n_epochs,
        checkpoint_path=checkpoint_path,
    )
    print(f"Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
