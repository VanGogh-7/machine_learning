import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, sort_keys=True)


def load_json(input_path: Path) -> dict[str, Any]:
    with input_path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def retrieval_accuracy_from_scores(scores: torch.Tensor) -> int:
    # The diagonal item is the positive item for each user in the batch.
    labels = torch.arange(scores.size(0), device=scores.device)
    predictions = scores.argmax(dim=1)
    return int((predictions == labels).sum().item())


def recall_at_k_from_scores(scores: torch.Tensor, k: int) -> int:
    # Recall@K is 1 for a row when the diagonal positive item is in its top K.
    if scores.numel() == 0:
        return 0
    labels = torch.arange(scores.size(0), device=scores.device)
    effective_k = min(k, scores.size(1))
    topk_indices = scores.topk(k=effective_k, dim=1).indices
    hits = (topk_indices == labels.unsqueeze(1)).any(dim=1)
    return int(hits.sum().item())
