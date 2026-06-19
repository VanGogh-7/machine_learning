from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _plot_metric(
    epochs: range,
    train_values: list[float | None],
    valid_values: list[float | None],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    train_clean = [value for value in train_values if value is not None]
    valid_clean = [value for value in valid_values if value is not None]
    if not train_clean or not valid_clean:
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
        title="NCF Loss",
        ylabel="BCEWithLogitsLoss",
        output_path=output_dir / "loss_curve.png",
    )
    _plot_metric(
        epochs=epochs,
        train_values=history.get("train_accuracies", []),
        valid_values=history.get("valid_accuracies", []),
        title="NCF Accuracy",
        ylabel="Accuracy",
        output_path=output_dir / "accuracy_curve.png",
    )
    _plot_metric(
        epochs=epochs,
        train_values=history.get("train_aucs", []),
        valid_values=history.get("valid_aucs", []),
        title="NCF AUC",
        ylabel="AUC",
        output_path=output_dir / "auc_curve.png",
    )
