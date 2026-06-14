from collections.abc import Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.metrics import top_k_accuracy
from src.utils import save_checkpoint


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a model and return average loss, top-1, and top-5 accuracy."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_top1 += top_k_accuracy(logits, targets, k=1).item() * batch_size
        total_top5 += top_k_accuracy(logits, targets, k=5).item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch and return average loss and top-1 accuracy."""
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_top1 += top_k_accuracy(logits.detach(), targets, k=1).item() * batch_size
        total_samples += batch_size

    return {"loss": total_loss / total_samples, "top1": total_top1 / total_samples}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    n_epochs: int,
    model_path: str,
) -> dict[str, list[float]]:
    """Run training, save the best validation checkpoint, and return metric history."""
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_top1": [],
        "valid_top1": [],
        "valid_top5": [],
    }
    best_valid_top1 = -1.0

    for epoch in range(1, n_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_metrics = evaluate(model, valid_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_top1"].append(train_metrics["top1"])
        history["valid_top1"].append(valid_metrics["top1"])
        history["valid_top5"].append(valid_metrics["top5"])

        if valid_metrics["top1"] > best_valid_top1:
            best_valid_top1 = valid_metrics["top1"]
            save_checkpoint(model, model_path)

        print(
            f"Epoch {epoch:02d}/{n_epochs:02d} | "
            f"train loss: {train_metrics['loss']:.4f} | "
            f"train top-1: {train_metrics['top1']:.2%} | "
            f"valid top-1: {valid_metrics['top1']:.2%} | "
            f"valid top-5: {valid_metrics['top5']:.2%}"
        )

    return history
