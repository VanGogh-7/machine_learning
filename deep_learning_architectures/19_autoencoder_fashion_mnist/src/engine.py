from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils import save_checkpoint, save_json


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, _ in data_loader:
        images = images.to(device)

        optimizer.zero_grad()
        reconstructed_images = model(images)
        loss = loss_fn(reconstructed_images, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            reconstructed_images = model(images)
            loss = loss_fn(reconstructed_images, images)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return total_loss / total_samples


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
        "valid_losses": [],
    }
    best_valid_loss = float("inf")

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        valid_loss = evaluate(
            model=model,
            data_loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history["train_losses"].append(train_loss)
        history["valid_losses"].append(valid_loss)
        save_json(history, history_path)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model, checkpoint_path)
            checkpoint_status = "saved best checkpoint"
        else:
            checkpoint_status = "checkpoint unchanged"

        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train reconstruction loss: {train_loss:.6f}, "
            f"valid reconstruction loss: {valid_loss:.6f}, "
            f"{checkpoint_status}"
        )

    return history

