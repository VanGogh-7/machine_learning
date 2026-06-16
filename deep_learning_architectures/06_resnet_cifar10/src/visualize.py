import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import CIFAR10_MEAN, CIFAR10_STD


def unnormalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def show_image(image: torch.Tensor, title: str | None = None) -> None:
    display_image = unnormalize_image(image.detach().cpu()).permute(1, 2, 0)
    plt.imshow(display_image)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()


def plot_learning_curves(history: dict[str, list[float]]) -> None:
    epochs = np.arange(len(history["train_accuracies"])) + 1
    plt.plot(epochs, history["train_accuracies"], ".--", label="Training")
    plt.plot(epochs, history["valid_accuracies"], ".-", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.title("ResNet CIFAR-10 learning curves")
    plt.legend()
    plt.show()
