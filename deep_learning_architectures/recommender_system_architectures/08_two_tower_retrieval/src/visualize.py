from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _plot_metric(
    epochs: range,
    train_values: list[float],
    valid_values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if not train_values or not valid_values:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_values, marker="o", label="Train")
    plt.plot(epochs, valid_values, marker="o", label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_training_curves(history: dict[str, list[Any]], output_dir: Path) -> None:
    epochs = range(1, len(history.get("train_losses", [])) + 1)
    if not epochs:
        return

    _plot_metric(
        epochs=epochs,
        train_values=history.get("train_losses", []),
        valid_values=history.get("valid_losses", []),
        title="Two-Tower Retrieval Loss",
        ylabel="CrossEntropyLoss",
        output_path=output_dir / "loss_curve.png",
    )
    _plot_metric(
        epochs=epochs,
        train_values=history.get("train_retrieval_accuracies", []),
        valid_values=history.get("valid_retrieval_accuracies", []),
        title="Two-Tower Retrieval Accuracy",
        ylabel="Retrieval Accuracy",
        output_path=output_dir / "retrieval_accuracy_curve.png",
    )
    for k in (1, 5, 10):
        _plot_metric(
            epochs=epochs,
            train_values=history.get(f"train_recall_at_{k}", []),
            valid_values=history.get(f"valid_recall_at_{k}", []),
            title=f"Two-Tower Recall@{k}",
            ylabel=f"Recall@{k}",
            output_path=output_dir / f"recall_at_{k}_curve.png",
        )
