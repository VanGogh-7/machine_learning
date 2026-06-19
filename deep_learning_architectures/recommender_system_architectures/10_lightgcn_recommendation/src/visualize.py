from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _plot_values(
    values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if not values:
        return
    epochs = range(1, len(values) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_training_curves(history: dict[str, list[Any]], output_dir: Path) -> None:
    _plot_values(
        values=history.get("train_bpr_losses", []),
        title="LightGCN BPR Loss",
        ylabel="BPR Loss",
        output_path=output_dir / "bpr_loss_curve.png",
    )
    for key, values in history.items():
        if key.startswith("valid_recall_at_"):
            _plot_values(
                values=values,
                title=f"LightGCN {key.replace('_', ' ').title()}",
                ylabel="Recall",
                output_path=output_dir / f"{key}_curve.png",
            )
        if key.startswith("valid_ndcg_at_"):
            _plot_values(
                values=values,
                title=f"LightGCN {key.replace('_', ' ').title()}",
                ylabel="NDCG",
                output_path=output_dir / f"{key}_curve.png",
            )
