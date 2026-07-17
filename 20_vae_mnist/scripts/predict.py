import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_datasets
from src.engine import vae_loss_function
from src.model import ConvVAE
from src.utils import get_device, load_checkpoint
from src.visualize import save_generated_grid, save_reconstruction_grid


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    _, _, test_dataset = build_datasets(
        data_root=config.data_root,
        valid_ratio=config.valid_ratio,
        seed=config.seed,
    )
    sample_count = min(8, len(test_dataset))
    sample_loader = DataLoader(
        Subset(test_dataset, range(sample_count)),
        batch_size=sample_count,
        shuffle=False,
    )

    model = ConvVAE(
        in_channels=config.in_channels,
        latent_dim=config.latent_dim,
        base_channels=config.base_channels,
        image_size=config.image_size,
    ).to(device)
    load_checkpoint(model, PROJECT_ROOT / config.model_path, device)
    model.eval()

    images, _ = next(iter(sample_loader))
    images = images.to(device)
    with torch.no_grad():
        reconstructed_images, mu, logvar = model(images)
        total_loss, reconstruction_loss, kl_loss = vae_loss_function(
            reconstructed_images,
            images,
            mu,
            logvar,
            config.beta,
        )
        z = torch.randn(config.num_generation_samples, config.latent_dim, device=device)
        generated_images = model.decode(z)

    output_dir = PROJECT_ROOT / config.output_dir
    save_reconstruction_grid(
        images.cpu(),
        reconstructed_images.cpu(),
        output_dir / "reconstructions.png",
    )
    save_generated_grid(generated_images.cpu(), output_dir / "generated_samples.png")
    print(f"Selected batch total loss: {total_loss.item() / images.size(0):.4f}")
    print(
        "Selected batch reconstruction loss: "
        f"{reconstruction_loss.item() / images.size(0):.4f}"
    )
    print(f"Selected batch KL loss: {kl_loss.item() / images.size(0):.4f}")
    print(f"Saved reconstruction comparison to {output_dir / 'reconstructions.png'}")
    print(f"Saved generated samples to {output_dir / 'generated_samples.png'}")


if __name__ == "__main__":
    main()

