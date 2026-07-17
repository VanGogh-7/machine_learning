from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    return ((images.detach().cpu() + 1.0) / 2.0).clamp(0, 1)


def save_generated_grid(
    images: torch.Tensor,
    output_path: Path,
    max_images: int = 64,
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
            image = images[index].permute(1, 2, 0).numpy()
            axes[row, col].imshow(image)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Path | None = None,
) -> None:
    epochs = np.arange(len(history["discriminator_losses"])) + 1
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["discriminator_losses"], label="Discriminator")
    axes[0].plot(epochs, history["generator_losses"], label="Generator")
    axes[0].set_title("DCGAN Losses")

    axes[1].plot(epochs, history["real_accuracies"], label="Real accuracy")
    axes[1].plot(epochs, history["fake_accuracies"], label="Fake accuracy")
    axes[1].set_title("Discriminator Accuracy")

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid()
        axis.legend()
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()

