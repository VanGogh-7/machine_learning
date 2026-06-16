import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloader
from src.engine import train
from src.model import Discriminator, Generator
from src.utils import clear_memory, get_device, make_fixed_noise, set_seed


def main() -> None:
    config = TrainConfig()
    clear_memory()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader = build_dataloader(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    print(f"Train size: {len(train_loader.dataset)}")

    generator = Generator(
        latent_dim=config.latent_dim,
        image_dim=config.image_dim,
        hidden_dims=config.generator_hidden_dims,
        in_channels=config.in_channels,
        image_size=config.image_size,
    ).to(device)
    discriminator = Discriminator(
        image_dim=config.image_dim,
        hidden_dims=config.discriminator_hidden_dims,
    ).to(device)

    real_images, _ = next(iter(train_loader))
    noise = torch.randn(real_images.size(0), config.latent_dim, device=device)
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        generated_images = generator(noise)
        discriminator_logits = discriminator(generated_images)
    print(f"Real images shape: {tuple(real_images.shape)}")
    print(f"Noise shape: {tuple(noise.shape)}")
    print(f"Generated images shape: {tuple(generated_images.shape)}")
    print(f"Discriminator output shape: {tuple(discriminator_logits.shape)}")

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate_g,
        betas=(config.beta1, config.beta2),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate_d,
        betas=(config.beta1, config.beta2),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    fixed_noise = make_fixed_noise(
        num_samples=config.num_generation_samples,
        latent_dim=config.latent_dim,
        seed=config.seed,
        device=device,
    )

    train(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        loss_fn=loss_fn,
        device=device,
        latent_dim=config.latent_dim,
        n_epochs=config.n_epochs,
        generator_path=PROJECT_ROOT / config.generator_path,
        discriminator_path=PROJECT_ROOT / config.discriminator_path,
        history_path=PROJECT_ROOT / config.output_dir / config.history_path,
        output_dir=PROJECT_ROOT / config.output_dir,
        fixed_noise=fixed_noise,
        sample_interval=config.sample_interval,
    )


if __name__ == "__main__":
    main()

