from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_reconstruction_losses(
    history: dict[str, list[float]],
    output_path: Path | None = None,
) -> None:
    epochs = np.arange(len(history["train_losses"])) + 1
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_losses"], label="Training")
    plt.plot(epochs, history["valid_losses"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss")
    plt.title("AutoEncoder Reconstruction Loss")
    plt.grid()
    plt.legend()
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

