import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.diffusion import GaussianDiffusion
from src.model import SimpleUNetNoisePredictor
from src.utils import get_device, load_checkpoint, set_seed
from src.visualize import save_image_grid


def main() -> None:
    config = TrainConfig()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    model = SimpleUNetNoisePredictor(
        image_channels=config.image_channels,
        base_channels=config.base_channels,
        time_embedding_dim=config.time_embedding_dim,
    ).to(device)
    load_checkpoint(model, PROJECT_ROOT / config.model_path, device)
    diffusion = GaussianDiffusion(
        num_timesteps=config.num_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )

    shape = (
        config.num_generation_samples,
        config.image_channels,
        config.image_size,
        config.image_size,
    )
    samples = diffusion.sample(model=model, shape=shape, device=device)
    output_dir = PROJECT_ROOT / config.output_dir
    save_image_grid(samples, output_dir / "generated_samples.png")
    print(f"Saved generated samples to {output_dir / 'generated_samples.png'}")

    capture_steps = [config.num_timesteps, 750, 500, 250, 0]
    capture_steps = [step for step in capture_steps if step <= config.num_timesteps]
    intermediates = diffusion.sample_with_intermediates(
        model=model,
        shape=(1, config.image_channels, config.image_size, config.image_size),
        device=device,
        capture_steps=capture_steps,
    )
    for step, images in intermediates.items():
        output_path = output_dir / f"denoising_step_{step:04d}.png"
        save_image_grid(images, output_path, max_images=1, title=f"t={step}")
        print(f"Saved denoising step {step} to {output_path}")


if __name__ == "__main__":
    main()

