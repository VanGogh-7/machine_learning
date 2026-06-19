from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils import (
    accuracy_from_logits,
    binary_auc,
    save_checkpoint,
    save_json,
)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> dict[str, float | None]:
    model.train()
    total_loss_sum = 0.0
    total_correct = 0
    total_samples = 0
    probabilities = []
    all_targets = []

    for input_item_ids, target_item_ids, targets in tqdm(data_loader, leave=False):
        input_item_ids = input_item_ids.to(device)
        target_item_ids = target_item_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(input_item_ids, target_item_ids)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss_sum += loss.item() * batch_size
        total_correct += accuracy_from_logits(logits, targets)
        total_samples += batch_size
        probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    return {
        "loss": total_loss_sum / total_samples,
        "accuracy": total_correct / total_samples,
        "auc": binary_auc(probabilities, all_targets),
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> dict[str, float | None]:
    model.eval()
    total_loss_sum = 0.0
    total_correct = 0
    total_samples = 0
    probabilities = []
    all_targets = []

    with torch.no_grad():
        for input_item_ids, target_item_ids, targets in data_loader:
            input_item_ids = input_item_ids.to(device)
            target_item_ids = target_item_ids.to(device)
            targets = targets.to(device)

            logits = model(input_item_ids, target_item_ids)
            loss = loss_fn(logits, targets)

            batch_size = targets.size(0)
            total_loss_sum += loss.item() * batch_size
            total_correct += accuracy_from_logits(logits, targets)
            total_samples += batch_size
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    return {
        "loss": total_loss_sum / total_samples,
        "accuracy": total_correct / total_samples,
        "auc": binary_auc(probabilities, all_targets),
    }


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
) -> dict[str, list[float | None]]:
    history: dict[str, list[float | None]] = {
        "train_losses": [],
        "train_accuracies": [],
        "train_aucs": [],
        "valid_losses": [],
        "valid_accuracies": [],
        "valid_aucs": [],
    }
    best_valid_auc = -1.0
    best_valid_loss = float("inf")

    for epoch in range(n_epochs):
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        valid_metrics = evaluate(
            model=model,
            data_loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history["train_losses"].append(train_metrics["loss"])
        history["train_accuracies"].append(train_metrics["accuracy"])
        history["train_aucs"].append(train_metrics["auc"])
        history["valid_losses"].append(valid_metrics["loss"])
        history["valid_accuracies"].append(valid_metrics["accuracy"])
        history["valid_aucs"].append(valid_metrics["auc"])
        save_json(history, history_path)

        valid_auc = valid_metrics["auc"]
        valid_loss = float(valid_metrics["loss"])
        improved = (
            valid_auc is not None and valid_auc > best_valid_auc
        ) or (
            valid_auc is None and valid_loss < best_valid_loss
        )
        if improved:
            best_valid_auc = valid_auc if valid_auc is not None else best_valid_auc
            best_valid_loss = valid_loss
            save_checkpoint(model, checkpoint_path)
            checkpoint_status = "saved best checkpoint"
        else:
            checkpoint_status = "checkpoint unchanged"

        train_auc_text = (
            "n/a" if train_metrics["auc"] is None else f"{train_metrics['auc']:.4f}"
        )
        valid_auc_text = (
            "n/a" if valid_metrics["auc"] is None else f"{valid_metrics['auc']:.4f}"
        )
        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {train_metrics['loss']:.4f}, "
            f"train accuracy: {train_metrics['accuracy']:.4f}, "
            f"train AUC: {train_auc_text}, "
            f"valid loss: {valid_metrics['loss']:.4f}, "
            f"valid accuracy: {valid_metrics['accuracy']:.4f}, "
            f"valid AUC: {valid_auc_text}, "
            f"{checkpoint_status}"
        )

    return history
