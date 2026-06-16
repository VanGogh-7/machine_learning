from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_loss_curves(
    history: dict[str, list[float]],
    output_path: Path | None = None,
) -> None:
    epochs = np.arange(len(history["train_total_losses"])) + 1
    _, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, history["train_total_losses"], label="Training")
    axes[0].plot(epochs, history["valid_total_losses"], label="Validation")
    axes[0].set_title("Total Loss")

    axes[1].plot(epochs, history["train_reconstruction_losses"], label="Training")
    axes[1].plot(epochs, history["valid_reconstruction_losses"], label="Validation")
    axes[1].set_title("Reconstruction Loss")

    axes[2].plot(epochs, history["train_kl_losses"], label="Training")
    axes[2].plot(epochs, history["valid_kl_losses"], label="Validation")
    axes[2].set_title("KL Loss")

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


def save_reconstruction_grid(
    images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    output_path: Path,
    max_images: int = 8,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu()[:max_images]
    reconstructed_images = reconstructed_images.detach().cpu()[:max_images]
    n_images = images.size(0)

    _, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 4))
    if n_images == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for index in range(n_images):
        axes[0, index].imshow(images[index].squeeze(0), cmap="gray", vmin=0, vmax=1)
        axes[0, index].set_title("Original")
        axes[0, index].axis("off")

        axes[1, index].imshow(
            reconstructed_images[index].squeeze(0),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[1, index].set_title("Reconstruction")
        axes[1, index].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_generated_grid(
    generated_images: torch.Tensor,
    output_path: Path,
    max_images: int = 16,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_images = generated_images.detach().cpu()[:max_images]
    n_images = generated_images.size(0)
    grid_size = int(np.ceil(np.sqrt(n_images)))

    _, axes = plt.subplots(grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size))
    axes = np.array(axes).reshape(grid_size, grid_size)

    for index in range(grid_size * grid_size):
        row = index // grid_size
        col = index % grid_size
        axes[row, col].axis("off")
        if index < n_images:
            axes[row, col].imshow(
                generated_images[index].squeeze(0),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row, col].set_title("Generated")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

