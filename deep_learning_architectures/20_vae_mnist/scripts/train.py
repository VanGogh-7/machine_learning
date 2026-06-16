import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.engine import evaluate, train
from src.model import ConvVAE
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
        valid_ratio=config.valid_ratio,
        seed=config.seed,
    )
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(valid_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")

    model = ConvVAE(
        in_channels=config.in_channels,
        latent_dim=config.latent_dim,
        base_channels=config.base_channels,
        image_size=config.image_size,
    ).to(device)
    images, _ = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        reconstructed_images, mu, logvar = model(images.to(device))
    print(f"Images shape: {tuple(images.shape)}")
    print(f"Mu shape: {tuple(mu.shape)}")
    print(f"Logvar shape: {tuple(logvar.shape)}")
    print(f"Reconstructed images shape: {tuple(reconstructed_images.shape)}")

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
        device=device,
        beta=config.beta,
        n_epochs=config.n_epochs,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    load_checkpoint(model, checkpoint_path, device)
    test_total, test_reconstruction, test_kl = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        beta=config.beta,
    )
    print(f"Best model saved to {checkpoint_path}")
    print(f"Training history saved to {history_path}")
    print(f"Final test total loss: {test_total:.4f}")
    print(f"Final test reconstruction loss: {test_reconstruction:.4f}")
    print(f"Final test KL loss: {test_kl:.4f}")


if __name__ == "__main__":
    main()

