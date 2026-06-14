import torch


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Return the fraction of targets found among each sample's top-k predictions."""
    if k < 1 or k > logits.shape[1]:
        raise ValueError(f"k must be between 1 and {logits.shape[1]}.")
    predictions = logits.topk(k, dim=1).indices
    correct = predictions.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean()
