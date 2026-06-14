import matplotlib.pyplot as plt


def plot_learning_curves(history: dict[str, list[float]]) -> None:
    """Plot loss and accuracy metrics collected during training."""
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Training loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_top1"], label="Training top-1")
    axes[1].plot(epochs, history["valid_top1"], label="Validation top-1")
    axes[1].plot(epochs, history["valid_top5"], label="Validation top-5")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    plt.show()
