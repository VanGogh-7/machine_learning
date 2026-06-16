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


def masked_auxiliary_loss(
    aux_logits: torch.Tensor,
    aux_mask: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    # This educational auxiliary loss treats observed next history items as
    # positive examples and ignores padded or invalid positions.
    aux_targets = torch.ones_like(aux_logits)
    per_position_loss = loss_fn(aux_logits, aux_targets)
    valid_count = aux_mask.sum().clamp_min(1.0)
    return (per_position_loss * aux_mask).sum() / valid_count


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ctr_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    aux_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    aux_loss_weight: float,
) -> dict[str, float | None]:
    model.train()
    total_loss_sum = 0.0
    ctr_loss_sum = 0.0
    aux_loss_sum = 0.0
    total_correct = 0
    total_samples = 0
    probabilities = []
    all_targets = []

    for (
        target_items,
        histories,
        next_histories,
        history_masks,
        aux_masks,
        targets,
    ) in tqdm(data_loader, leave=False):
        target_items = target_items.to(device)
        histories = histories.to(device)
        next_histories = next_histories.to(device)
        history_masks = history_masks.to(device)
        aux_masks = aux_masks.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(
            target_item_ids=target_items,
            history_item_ids=histories,
            next_history_item_ids=next_histories,
            history_mask=history_masks,
            aux_mask=aux_masks,
        )
        ctr_loss = ctr_loss_fn(outputs["logits"], targets)
        aux_loss = masked_auxiliary_loss(
            outputs["aux_logits"],
            outputs["aux_mask"],
            aux_loss_fn,
        )
        total_loss = ctr_loss + aux_loss_weight * aux_loss
        total_loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss_sum += total_loss.item() * batch_size
        ctr_loss_sum += ctr_loss.item() * batch_size
        aux_loss_sum += aux_loss.item() * batch_size
        total_correct += accuracy_from_logits(outputs["logits"], targets)
        total_samples += batch_size
        probabilities.extend(torch.sigmoid(outputs["logits"]).detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    return {
        "total_loss": total_loss_sum / total_samples,
        "ctr_loss": ctr_loss_sum / total_samples,
        "aux_loss": aux_loss_sum / total_samples,
        "accuracy": total_correct / total_samples,
        "auc": binary_auc(probabilities, all_targets),
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    ctr_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    aux_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    aux_loss_weight: float,
) -> dict[str, float | None]:
    model.eval()
    total_loss_sum = 0.0
    ctr_loss_sum = 0.0
    aux_loss_sum = 0.0
    total_correct = 0
    total_samples = 0
    probabilities = []
    all_targets = []

    with torch.no_grad():
        for (
            target_items,
            histories,
            next_histories,
            history_masks,
            aux_masks,
            targets,
        ) in data_loader:
            target_items = target_items.to(device)
            histories = histories.to(device)
            next_histories = next_histories.to(device)
            history_masks = history_masks.to(device)
            aux_masks = aux_masks.to(device)
            targets = targets.to(device)

            outputs = model(
                target_item_ids=target_items,
                history_item_ids=histories,
                next_history_item_ids=next_histories,
                history_mask=history_masks,
                aux_mask=aux_masks,
            )
            ctr_loss = ctr_loss_fn(outputs["logits"], targets)
            aux_loss = masked_auxiliary_loss(
                outputs["aux_logits"],
                outputs["aux_mask"],
                aux_loss_fn,
            )
            total_loss = ctr_loss + aux_loss_weight * aux_loss

            batch_size = targets.size(0)
            total_loss_sum += total_loss.item() * batch_size
            ctr_loss_sum += ctr_loss.item() * batch_size
            aux_loss_sum += aux_loss.item() * batch_size
            total_correct += accuracy_from_logits(outputs["logits"], targets)
            total_samples += batch_size
            probabilities.extend(torch.sigmoid(outputs["logits"]).cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    return {
        "total_loss": total_loss_sum / total_samples,
        "ctr_loss": ctr_loss_sum / total_samples,
        "aux_loss": aux_loss_sum / total_samples,
        "accuracy": total_correct / total_samples,
        "auc": binary_auc(probabilities, all_targets),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ctr_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    aux_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    aux_loss_weight: float,
    n_epochs: int,
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, list[float | None]]:
    history: dict[str, list[float | None]] = {
        "train_total_losses": [],
        "train_ctr_losses": [],
        "train_aux_losses": [],
        "train_accuracies": [],
        "train_aucs": [],
        "valid_total_losses": [],
        "valid_ctr_losses": [],
        "valid_aux_losses": [],
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
            ctr_loss_fn=ctr_loss_fn,
            aux_loss_fn=aux_loss_fn,
            device=device,
            aux_loss_weight=aux_loss_weight,
        )
        valid_metrics = evaluate(
            model=model,
            data_loader=valid_loader,
            ctr_loss_fn=ctr_loss_fn,
            aux_loss_fn=aux_loss_fn,
            device=device,
            aux_loss_weight=aux_loss_weight,
        )

        history["train_total_losses"].append(train_metrics["total_loss"])
        history["train_ctr_losses"].append(train_metrics["ctr_loss"])
        history["train_aux_losses"].append(train_metrics["aux_loss"])
        history["train_accuracies"].append(train_metrics["accuracy"])
        history["train_aucs"].append(train_metrics["auc"])
        history["valid_total_losses"].append(valid_metrics["total_loss"])
        history["valid_ctr_losses"].append(valid_metrics["ctr_loss"])
        history["valid_aux_losses"].append(valid_metrics["aux_loss"])
        history["valid_accuracies"].append(valid_metrics["accuracy"])
        history["valid_aucs"].append(valid_metrics["auc"])
        save_json(history, history_path)

        valid_auc = valid_metrics["auc"]
        valid_loss = float(valid_metrics["total_loss"])
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
            f"train total loss: {train_metrics['total_loss']:.4f}, "
            f"train CTR loss: {train_metrics['ctr_loss']:.4f}, "
            f"train aux loss: {train_metrics['aux_loss']:.4f}, "
            f"train accuracy: {train_metrics['accuracy']:.4f}, "
            f"train AUC: {train_auc_text}, "
            f"valid total loss: {valid_metrics['total_loss']:.4f}, "
            f"valid CTR loss: {valid_metrics['ctr_loss']:.4f}, "
            f"valid aux loss: {valid_metrics['aux_loss']:.4f}, "
            f"valid accuracy: {valid_metrics['accuracy']:.4f}, "
            f"valid AUC: {valid_auc_text}, "
            f"{checkpoint_status}"
        )

    return history
