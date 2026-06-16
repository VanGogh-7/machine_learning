import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_learning_curves(history: dict[str, list[float]]) -> None:
    epochs = np.arange(len(history["train_losses"])) + 1
    plt.plot(epochs, history["train_losses"], ".--", label="Training")
    plt.plot(epochs, history["valid_losses"], ".-", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.title("Seq2Seq attention learning curves")
    plt.legend()
    plt.show()


def plot_attention(
    attention_weights: torch.Tensor,
    source_tokens: list[str],
    target_tokens: list[str],
) -> None:
    weights = attention_weights.detach().cpu().numpy()
    _, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(weights, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(source_tokens)), source_tokens, rotation=45, ha="right")
    axis.set_yticks(range(len(target_tokens)), target_tokens)
    axis.set_xlabel("Source tokens")
    axis.set_ylabel("Generated target tokens")
    plt.colorbar(image)
    plt.tight_layout()
    plt.show()
