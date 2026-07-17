import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(history: dict[str, list[float]]) -> None:
    epochs = np.arange(len(history["train_accuracies"])) + 1
    plt.plot(epochs, history["train_accuracies"], ".--", label="Training")
    plt.plot(epochs, history["valid_accuracies"], ".-", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.title("LSTM text classification learning curves")
    plt.legend()
    plt.show()
