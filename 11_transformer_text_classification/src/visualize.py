import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(history: dict[str, list[float]]) -> None:
    epochs = np.arange(len(history["train_losses"])) + 1
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_losses"], ".--", label="Training")
    axes[0].plot(epochs, history["valid_losses"], ".-", label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracies"], ".--", label="Training")
    axes[1].plot(epochs, history["valid_accuracies"], ".-", label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid()
    axes[1].legend()

    plt.tight_layout()
    plt.show()
