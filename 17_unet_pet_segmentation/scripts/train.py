import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.engine import evaluate, train
from src.model import UNet
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
        valid_ratio=config.valid_ratio,
        image_size=config.image_size,
        use_augmentation=config.use_augmentation,
    )
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(valid_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")

    model = UNet(
        in_channels=3,
        num_classes=config.n_classes,
        base_channels=config.base_channels,
        dropout=config.dropout,
    ).to(device)
    images, masks = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
    print(f"Images shape: {tuple(images.shape)}")
    print(f"Masks shape: {tuple(masks.shape)}")
    print(f"Model output shape: {tuple(outputs.shape)}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_path = PROJECT_ROOT / config.model_path
    history_path = PROJECT_ROOT / config.output_dir / config.history_path

    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_classes=config.n_classes,
        n_epochs=config.n_epochs,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    load_checkpoint(model, checkpoint_path, device)
    test_loss, test_accuracy, test_iou = evaluate(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        num_classes=config.n_classes,
    )
    test_iou_text = "n/a" if test_iou is None else f"{test_iou:.4f}"
    print(f"Best model saved to {checkpoint_path}")
    print(f"Training history saved to {history_path}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test pixel accuracy: {test_accuracy:.4f}")
    print(f"Final test mean IoU: {test_iou_text}")


if __name__ == "__main__":
    main()
