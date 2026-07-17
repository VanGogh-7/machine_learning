from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def unnormalize_image(image: torch.Tensor) -> torch.Tensor:
    return (image.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def plot_training_curves(
    history: dict[str, list[float | None]],
    output_path: Path | None = None,
) -> None:
    epochs = np.arange(len(history["train_losses"])) + 1
    _, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, history["train_losses"], label="Training")
    axes[0].plot(epochs, history["valid_losses"], label="Validation")
    axes[0].set_title("Loss")

    axes[1].plot(epochs, history["train_pixel_accuracies"], label="Training")
    axes[1].plot(epochs, history["valid_pixel_accuracies"], label="Validation")
    axes[1].set_title("Pixel Accuracy")

    if all(value is not None for value in history["train_mean_ious"]):
        axes[2].plot(epochs, history["train_mean_ious"], label="Training")
    if all(value is not None for value in history["valid_mean_ious"]):
        axes[2].plot(epochs, history["valid_mean_ious"], label="Validation")
    axes[2].set_title("Mean IoU")

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


def save_segmentation_comparison(
    image: torch.Tensor,
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_np = unnormalize_image(image).permute(1, 2, 0).numpy()
    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Input image")
    axes[1].imshow(true_mask.cpu().numpy(), vmin=0, vmax=2, cmap="viridis")
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred_mask.cpu().numpy(), vmin=0, vmax=2, cmap="viridis")
    axes[2].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
