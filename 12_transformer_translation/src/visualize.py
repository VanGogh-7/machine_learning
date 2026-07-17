from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(
    history: dict[str, list[float]],
    output_path: Path | None = None,
) -> None:
    epochs = np.arange(len(history["train_losses"])) + 1
    plt.plot(epochs, history["train_losses"], ".--", label="Training")
    plt.plot(epochs, history["valid_losses"], ".-", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Token-level loss")
    plt.title("Transformer translation learning curves")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()
