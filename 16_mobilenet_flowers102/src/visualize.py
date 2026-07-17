from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
