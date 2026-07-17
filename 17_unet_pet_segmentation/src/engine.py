from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils import (
    intersection_and_union,
    mean_iou,
    pixel_accuracy,
    save_checkpoint,
    save_json,
)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, float | None]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    total_intersections = torch.zeros(num_classes, device=device)
    total_unions = torch.zeros(num_classes, device=device)

    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        batch_pixels = masks.numel()
        correct, pixels = pixel_accuracy(logits, masks)
        intersections, unions = intersection_and_union(logits, masks, num_classes)
        total_loss += loss.item() * batch_pixels
        total_correct += correct
        total_pixels += pixels
        total_intersections += intersections
        total_unions += unions

    return (
        total_loss / total_pixels,
        total_correct / total_pixels,
        mean_iou(total_intersections, total_unions),
    )


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, float | None]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    total_intersections = torch.zeros(num_classes, device=device)
    total_unions = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = loss_fn(logits, masks)

            batch_pixels = masks.numel()
            correct, pixels = pixel_accuracy(logits, masks)
            intersections, unions = intersection_and_union(logits, masks, num_classes)
            total_loss += loss.item() * batch_pixels
            total_correct += correct
            total_pixels += pixels
            total_intersections += intersections
            total_unions += unions

    return (
        total_loss / total_pixels,
        total_correct / total_pixels,
        mean_iou(total_intersections, total_unions),
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    num_classes: int,
    n_epochs: int,
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, list[float | None]]:
    history: dict[str, list[float | None]] = {
        "train_losses": [],
        "train_pixel_accuracies": [],
        "train_mean_ious": [],
        "valid_losses": [],
        "valid_pixel_accuracies": [],
        "valid_mean_ious": [],
    }
    best_valid_iou = -1.0
    best_valid_loss = float("inf")

    for epoch in range(n_epochs):
        train_loss, train_accuracy, train_iou = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, num_classes
        )
        valid_loss, valid_accuracy, valid_iou = evaluate(
            model, valid_loader, loss_fn, device, num_classes
        )

        history["train_losses"].append(train_loss)
        history["train_pixel_accuracies"].append(train_accuracy)
        history["train_mean_ious"].append(train_iou)
        history["valid_losses"].append(valid_loss)
        history["valid_pixel_accuracies"].append(valid_accuracy)
        history["valid_mean_ious"].append(valid_iou)
        save_json(history, history_path)

        improved = (
            valid_iou is not None and valid_iou > best_valid_iou
        ) or (
            valid_iou is None and valid_loss < best_valid_loss
        )
        if improved:
            best_valid_iou = valid_iou if valid_iou is not None else best_valid_iou
            best_valid_loss = valid_loss
            save_checkpoint(model, checkpoint_path)
            checkpoint_status = "saved best checkpoint"
        else:
            checkpoint_status = "checkpoint unchanged"

        train_iou_text = "n/a" if train_iou is None else f"{train_iou:.4f}"
        valid_iou_text = "n/a" if valid_iou is None else f"{valid_iou:.4f}"
        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {train_loss:.4f}, "
            f"train pixel accuracy: {train_accuracy:.4f}, "
            f"train mean IoU: {train_iou_text}, "
            f"valid loss: {valid_loss:.4f}, "
            f"valid pixel accuracy: {valid_accuracy:.4f}, "
            f"valid mean IoU: {valid_iou_text}, "
            f"{checkpoint_status}"
        )

    return history
