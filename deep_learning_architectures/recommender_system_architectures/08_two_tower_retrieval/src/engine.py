from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils import (
    recall_at_k_from_scores,
    retrieval_accuracy_from_scores,
    save_checkpoint,
    save_json,
)


RECALL_K_VALUES = (1, 5, 10)


def _batch_metrics(scores: torch.Tensor) -> dict[str, int]:
    metrics = {
        "correct": retrieval_accuracy_from_scores(scores),
    }
    for k in RECALL_K_VALUES:
        metrics[f"recall_at_{k}"] = recall_at_k_from_scores(scores, k)
    return metrics


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss_sum = 0.0
    total_correct = 0
    total_recall_hits = {k: 0 for k in RECALL_K_VALUES}
    total_samples = 0

    for user_ids, item_ids in tqdm(data_loader, leave=False):
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        labels = torch.arange(user_ids.size(0), device=device)

        optimizer.zero_grad()
        outputs = model(user_ids, item_ids)
        scores = outputs["scores"]
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()

        batch_size = user_ids.size(0)
        batch_metrics = _batch_metrics(scores.detach())
        total_loss_sum += loss.item() * batch_size
        total_correct += batch_metrics["correct"]
        for k in RECALL_K_VALUES:
            total_recall_hits[k] += batch_metrics[f"recall_at_{k}"]
        total_samples += batch_size

    metrics = {
        "loss": total_loss_sum / total_samples,
        "retrieval_accuracy": total_correct / total_samples,
    }
    for k in RECALL_K_VALUES:
        metrics[f"recall_at_{k}"] = total_recall_hits[k] / total_samples
    return metrics


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss_sum = 0.0
    total_correct = 0
    total_recall_hits = {k: 0 for k in RECALL_K_VALUES}
    total_samples = 0

    with torch.no_grad():
        for user_ids, item_ids in data_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            labels = torch.arange(user_ids.size(0), device=device)

            outputs = model(user_ids, item_ids)
            scores = outputs["scores"]
            loss = loss_fn(scores, labels)

            batch_size = user_ids.size(0)
            batch_metrics = _batch_metrics(scores)
            total_loss_sum += loss.item() * batch_size
            total_correct += batch_metrics["correct"]
            for k in RECALL_K_VALUES:
                total_recall_hits[k] += batch_metrics[f"recall_at_{k}"]
            total_samples += batch_size

    metrics = {
        "loss": total_loss_sum / total_samples,
        "retrieval_accuracy": total_correct / total_samples,
    }
    for k in RECALL_K_VALUES:
        metrics[f"recall_at_{k}"] = total_recall_hits[k] / total_samples
    return metrics


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
    history: dict[str, list[float]] = {
        "train_losses": [],
        "train_retrieval_accuracies": [],
        "train_recall_at_1": [],
        "train_recall_at_5": [],
        "train_recall_at_10": [],
        "valid_losses": [],
        "valid_retrieval_accuracies": [],
        "valid_recall_at_1": [],
        "valid_recall_at_5": [],
        "valid_recall_at_10": [],
    }
    best_valid_recall_at_10 = -1.0
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
        history["train_retrieval_accuracies"].append(
            train_metrics["retrieval_accuracy"]
        )
        history["train_recall_at_1"].append(train_metrics["recall_at_1"])
        history["train_recall_at_5"].append(train_metrics["recall_at_5"])
        history["train_recall_at_10"].append(train_metrics["recall_at_10"])
        history["valid_losses"].append(valid_metrics["loss"])
        history["valid_retrieval_accuracies"].append(
            valid_metrics["retrieval_accuracy"]
        )
        history["valid_recall_at_1"].append(valid_metrics["recall_at_1"])
        history["valid_recall_at_5"].append(valid_metrics["recall_at_5"])
        history["valid_recall_at_10"].append(valid_metrics["recall_at_10"])
        save_json(history, history_path)

        valid_recall_at_10 = valid_metrics["recall_at_10"]
        valid_loss = valid_metrics["loss"]
        improved = (
            valid_recall_at_10 > best_valid_recall_at_10
            or (
                valid_recall_at_10 == best_valid_recall_at_10
                and valid_loss < best_valid_loss
            )
        )
        if improved:
            best_valid_recall_at_10 = valid_recall_at_10
            best_valid_loss = valid_loss
            save_checkpoint(model, checkpoint_path)
            checkpoint_status = "saved best checkpoint"
        else:
            checkpoint_status = "checkpoint unchanged"

        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {train_metrics['loss']:.4f}, "
            f"train retrieval accuracy: {train_metrics['retrieval_accuracy']:.4f}, "
            f"train Recall@10: {train_metrics['recall_at_10']:.4f}, "
            f"valid loss: {valid_metrics['loss']:.4f}, "
            f"valid retrieval accuracy: {valid_metrics['retrieval_accuracy']:.4f}, "
            f"valid Recall@10: {valid_metrics['recall_at_10']:.4f}, "
            f"{checkpoint_status}"
        )

    return history
