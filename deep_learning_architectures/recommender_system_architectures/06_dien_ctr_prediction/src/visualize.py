from pathlib import Path

import matplotlib.pyplot as plt

from src.utils import load_json


def plot_training_curves(history_path: Path, output_dir: Path) -> Path:
    history = load_json(history_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "training_curves.png"

    epochs = range(1, len(history.get("train_total_losses", [])) + 1)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, history.get("train_total_losses", []), label="train total")
    axes[0, 0].plot(epochs, history.get("valid_total_losses", []), label="valid total")
    axes[0, 0].plot(epochs, history.get("train_ctr_losses", []), label="train CTR")
    axes[0, 0].plot(epochs, history.get("valid_ctr_losses", []), label="valid CTR")
    axes[0, 0].set_title("Total and CTR Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history.get("train_aux_losses", []), label="train aux")
    axes[0, 1].plot(epochs, history.get("valid_aux_losses", []), label="valid aux")
    axes[0, 1].set_title("Auxiliary Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, history.get("train_accuracies", []), label="train")
    axes[1, 0].plot(epochs, history.get("valid_accuracies", []), label="valid")
    axes[1, 0].set_title("Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()

    train_aucs = history.get("train_aucs", [])
    valid_aucs = history.get("valid_aucs", [])
    if any(value is not None for value in train_aucs + valid_aucs):
        axes[1, 1].plot(epochs, train_aucs, label="train")
        axes[1, 1].plot(epochs, valid_aucs, label="valid")
    axes[1, 1].set_title("AUC")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return plot_path
