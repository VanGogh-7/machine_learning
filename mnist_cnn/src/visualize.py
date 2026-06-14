import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    history: dict[str, list[float]],
    n_epochs: int,
) -> None:
    plt.plot(
        np.arange(n_epochs) + 0.5,
        history["train_metrics"],
        ".--",
        label="Training",
    )
    plt.plot(
        np.arange(n_epochs) + 1.0,
        history["valid_metrics"],
        ".-",
        label="Validation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.title("Learning curves")
    plt.axis([0.5, n_epochs + 0.5, 0.0, 1.0])
    plt.legend()
    plt.show()
