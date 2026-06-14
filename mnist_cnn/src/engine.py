from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    metric: Metric,
    device: torch.device,
) -> float:
    model.eval()
    metric.reset()

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            metric.update(logits, targets)

    return metric.compute().item()


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    metric: Metric,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    metric.reset()
    total_loss = 0.0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        metric.update(logits, targets)

    mean_loss = total_loss / len(data_loader)
    accuracy = metric.compute().item()
    return mean_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    train_metric: Metric,
    valid_metric: Metric,
    device: torch.device,
    n_epochs: int,
) -> dict[str, list[float]]:
    history = {
        "train_losses": [],
        "train_metrics": [],
        "valid_metrics": [],
    }

    for epoch in range(n_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric=train_metric,
            device=device,
        )
        valid_accuracy = evaluate(
            model=model,
            data_loader=valid_loader,
            metric=valid_metric,
            device=device,
        )

        history["train_losses"].append(train_loss)
        history["train_metrics"].append(train_accuracy)
        history["valid_metrics"].append(valid_accuracy)

        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {train_loss:.4f}, "
            f"train accuracy: {train_accuracy:.4f}, "
            f"valid accuracy: {valid_accuracy:.4f}"
        )

    return history
