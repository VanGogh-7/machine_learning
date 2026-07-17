import gc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pixel_accuracy(logits: torch.Tensor, masks: torch.Tensor) -> tuple[int, int]:
    predictions = logits.argmax(dim=1)
    correct = (predictions == masks).sum().item()
    total = masks.numel()
    return correct, total


def intersection_and_union(
    logits: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = logits.argmax(dim=1)
    intersections = torch.zeros(num_classes, device=logits.device)
    unions = torch.zeros(num_classes, device=logits.device)
    for class_index in range(num_classes):
        pred_mask = predictions == class_index
        true_mask = masks == class_index
        intersections[class_index] = (pred_mask & true_mask).sum()
        unions[class_index] = (pred_mask | true_mask).sum()
    return intersections, unions


def mean_iou(
    intersections: torch.Tensor,
    unions: torch.Tensor,
) -> float | None:
    valid = unions > 0
    if not valid.any():
        return None
    return (intersections[valid] / unions[valid]).mean().item()


def save_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)


def save_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)


def load_json(input_path: Path) -> dict[str, Any]:
    with input_path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
