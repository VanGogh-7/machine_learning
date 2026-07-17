from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    return ((images.detach().cpu() + 1.0) / 2.0).clamp(0, 1)


def save_image_grid(
    images: torch.Tensor,
    output_path: Path,
    max_images: int = 64,
    title: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = denormalize_images(images)[:max_images]
    n_images = images.size(0)
    grid_size = int(np.ceil(np.sqrt(n_images)))

    _, axes = plt.subplots(grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size))
    axes = np.array(axes).reshape(grid_size, grid_size)
    for index in range(grid_size * grid_size):
        row = index // grid_size
        col = index % grid_size
        axes[row, col].axis("off")
        if index < n_images:
            axes[row, col].imshow(images[index].squeeze(0), cmap="gray", vmin=0, vmax=1)
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_loss_curves(
    history: dict[str, list[float]],
    output_path: Path | None = None,
) -> None:
    epochs = np.arange(len(history["train_losses"])) + 1
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_losses"], label="Training")
    plt.plot(epochs, history["valid_losses"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Noise prediction loss")
    plt.title("DDPM MSE Loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def save_forward_noising_grid(
    images_by_step: dict[int, torch.Tensor],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = sorted(images_by_step.keys())
    _, axes = plt.subplots(1, len(steps), figsize=(3 * len(steps), 3))
    if len(steps) == 1:
        axes = [axes]
    for axis, step in zip(axes, steps, strict=True):
        image = denormalize_images(images_by_step[step])[0].squeeze(0)
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"t={step}")
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

