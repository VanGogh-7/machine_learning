from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def unnormalize_image(image: torch.Tensor) -> torch.Tensor:
    return (image.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Path | None = None,
) -> None:
    epochs = np.arange(len(history["train_losses"])) + 1
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_losses"], label="Training")
    axes[0].plot(epochs, history["valid_losses"], label="Validation")
    axes[0].set_title("Loss")

    axes[1].plot(epochs, history["train_accuracies"], label="Training")
    axes[1].plot(epochs, history["valid_accuracies"], label="Validation")
    axes[1].set_title("Accuracy")

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


def save_prediction_grid(
    images: list[torch.Tensor],
    titles: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]

    for axis, image, title in zip(axes, images, titles, strict=True):
        image_np = unnormalize_image(image).permute(1, 2, 0).numpy()
        axis.imshow(image_np)
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

