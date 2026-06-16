from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils import accuracy, save_checkpoint, save_json


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += accuracy(logits, labels)
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += accuracy(logits, labels)
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    n_epochs: int,
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, list[float]]:
    history = {
        "train_losses": [],
        "train_accuracies": [],
        "valid_losses": [],
        "valid_accuracies": [],
    }
    best_valid_accuracy = -1.0

    for epoch in range(n_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        valid_loss, valid_accuracy = evaluate(
            model=model,
            data_loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history["train_losses"].append(train_loss)
        history["train_accuracies"].append(train_accuracy)
        history["valid_losses"].append(valid_loss)
        history["valid_accuracies"].append(valid_accuracy)
        save_json(history, history_path)

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            save_checkpoint(model, checkpoint_path)
            checkpoint_status = "saved best checkpoint"
        else:
            checkpoint_status = "checkpoint unchanged"

        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {train_loss:.4f}, "
            f"train accuracy: {train_accuracy:.4f}, "
            f"valid loss: {valid_loss:.4f}, "
            f"valid accuracy: {valid_accuracy:.4f}, "
            f"{checkpoint_status}"
        )

    return history

