import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_datasets
from src.model import ConvAutoEncoder
from src.utils import get_device, load_checkpoint
from src.visualize import save_reconstruction_grid


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

    model = ConvAutoEncoder(
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
        reconstructed_images = model(images)
        reconstruction_loss = nn.MSELoss()(reconstructed_images, images).item()

    output_path = PROJECT_ROOT / config.output_dir / "reconstructions.png"
    save_reconstruction_grid(images.cpu(), reconstructed_images.cpu(), output_path)
    print(f"Selected batch reconstruction loss: {reconstruction_loss:.6f}")
    print(f"Saved reconstruction comparison to {output_path}")


if __name__ == "__main__":
    main()

