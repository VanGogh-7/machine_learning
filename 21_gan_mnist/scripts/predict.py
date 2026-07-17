import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.model import Generator
from src.utils import get_device, load_checkpoint, make_fixed_noise, set_seed
from src.visualize import save_generated_grid


def main() -> None:
    config = TrainConfig()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    generator = Generator(
        latent_dim=config.latent_dim,
        image_dim=config.image_dim,
        hidden_dims=config.generator_hidden_dims,
        in_channels=config.in_channels,
        image_size=config.image_size,
    ).to(device)
    load_checkpoint(generator, PROJECT_ROOT / config.generator_path, device)
    generator.eval()

    random_noise = torch.randn(
        config.num_generation_samples,
        config.latent_dim,
        device=device,
    )
    fixed_noise = make_fixed_noise(
        num_samples=config.num_generation_samples,
        latent_dim=config.latent_dim,
        seed=config.seed,
        device=device,
    )
    with torch.no_grad():
        random_samples = generator(random_noise)
        fixed_samples = generator(fixed_noise)

    output_dir = PROJECT_ROOT / config.output_dir
    save_generated_grid(random_samples, output_dir / "generated_random.png")
    save_generated_grid(fixed_samples, output_dir / "generated_fixed.png")
    print(f"Saved random generated samples to {output_dir / 'generated_random.png'}")
    print(f"Saved fixed-noise generated samples to {output_dir / 'generated_fixed.png'}")


if __name__ == "__main__":
    main()

