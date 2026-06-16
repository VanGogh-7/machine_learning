import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.engine import evaluate, train
from src.model import ResNetCIFAR10
from src.utils import clear_memory, get_device, load_checkpoint, set_seed


def main() -> None:
    config = TrainConfig()
    clear_memory()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader = build_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        valid_size=config.valid_size,
    )
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(valid_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")

    model = ResNetCIFAR10(
        in_channels=config.in_channels,
        n_classes=config.n_classes,
    ).to(device)
    batch_inputs, batch_targets = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        batch_outputs = model(batch_inputs.to(device))
    print(f"Image batch shape: {tuple(batch_inputs.shape)}")
    print(f"Label batch shape: {tuple(batch_targets.shape)}")
    print(f"Model output shape: {tuple(batch_outputs.shape)}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_path = PROJECT_ROOT / config.model_path

    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        n_epochs=config.n_epochs,
        checkpoint_path=checkpoint_path,
    )

    load_checkpoint(model, checkpoint_path, device)
    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )
    print(f"Best model saved to {checkpoint_path}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
