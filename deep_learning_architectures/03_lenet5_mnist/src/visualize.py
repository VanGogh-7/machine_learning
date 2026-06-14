import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    history: dict[str, list[float]],
    n_epochs: int,
) -> None:
    epochs = np.arange(n_epochs) + 1
    plt.plot(epochs, history["train_accuracies"], ".--", label="Training")
    plt.plot(epochs, history["valid_accuracies"], ".-", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.title("LeNet-5 learning curves")
    plt.axis([0.5, n_epochs + 0.5, 0.0, 1.0])
    plt.legend()
    plt.show()
