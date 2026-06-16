import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.diffusion import GaussianDiffusion
from src.engine import evaluate, train
from src.model import SimpleUNetNoisePredictor
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

    model = SimpleUNetNoisePredictor(
        image_channels=config.image_channels,
        base_channels=config.base_channels,
        time_embedding_dim=config.time_embedding_dim,
    ).to(device)
    diffusion = GaussianDiffusion(
        num_timesteps=config.num_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    images, _ = next(iter(train_loader))
    images = images.to(device)
    timesteps = torch.randint(
        0,
        config.num_timesteps,
        (images.size(0),),
        device=device,
        dtype=torch.long,
    )
    noisy_images, _ = diffusion.q_sample(images, timesteps)
    model.eval()
    with torch.no_grad():
        predicted_noise = model(noisy_images, timesteps)
    print(f"Images shape: {tuple(images.shape)}")
    print(f"Timesteps shape: {tuple(timesteps.shape)}")
    print(f"Noisy images shape: {tuple(noisy_images.shape)}")
    print(f"Predicted noise shape: {tuple(predicted_noise.shape)}")

    checkpoint_path = PROJECT_ROOT / config.model_path
    history_path = PROJECT_ROOT / config.output_dir / config.history_path
    train(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        n_epochs=config.n_epochs,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        output_dir=PROJECT_ROOT / config.output_dir,
        sample_interval=config.sample_interval,
        num_generation_samples=config.num_generation_samples,
        image_channels=config.image_channels,
        image_size=config.image_size,
    )

    load_checkpoint(model, checkpoint_path, device)
    test_loss = evaluate(
        model=model,
        diffusion=diffusion,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )
    print(f"Best model saved to {checkpoint_path}")
    print(f"Training history saved to {history_path}")
    print(f"Final test noise prediction loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()

