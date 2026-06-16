from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils import save_checkpoint, token_accuracy


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    return loss_fn(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    pad_idx: int,
    teacher_forcing_ratio: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for src_ids, tgt_ids in data_loader:
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)
        tgt_input_ids = tgt_ids[:, :-1]
        tgt_output_ids = tgt_ids[:, 1:]

        optimizer.zero_grad()
        logits = model(src_ids, tgt_input_ids, teacher_forcing_ratio)
        loss = compute_loss(logits, tgt_output_ids, loss_fn)
        loss.backward()
        optimizer.step()

        non_pad_tokens = (tgt_output_ids != pad_idx).sum().item()
        correct, token_count = token_accuracy(logits, tgt_output_ids, pad_idx)
        total_loss += loss.item() * non_pad_tokens
        total_correct += correct
        total_tokens += token_count

    return total_loss / total_tokens, total_correct / total_tokens


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    pad_idx: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for src_ids, tgt_ids in data_loader:
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)
            tgt_input_ids = tgt_ids[:, :-1]
            tgt_output_ids = tgt_ids[:, 1:]
            logits = model(src_ids, tgt_input_ids, teacher_forcing_ratio=0.0)
            loss = compute_loss(logits, tgt_output_ids, loss_fn)

            non_pad_tokens = (tgt_output_ids != pad_idx).sum().item()
            correct, token_count = token_accuracy(logits, tgt_output_ids, pad_idx)
            total_loss += loss.item() * non_pad_tokens
            total_correct += correct
            total_tokens += token_count

    return total_loss / total_tokens, total_correct / total_tokens


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    pad_idx: int,
    teacher_forcing_ratio: float,
    n_epochs: int,
    checkpoint_path: Path,
) -> dict[str, list[float]]:
    history = {
        "train_losses": [],
        "train_accuracies": [],
        "valid_losses": [],
        "valid_accuracies": [],
    }
    best_valid_loss = float("inf")

    for epoch in range(n_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            pad_idx=pad_idx,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        valid_loss, valid_accuracy = evaluate(
            model=model,
            data_loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
            pad_idx=pad_idx,
        )

        history["train_losses"].append(train_loss)
        history["train_accuracies"].append(train_accuracy)
        history["valid_losses"].append(valid_loss)
        history["valid_accuracies"].append(valid_accuracy)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model, checkpoint_path)
            checkpoint_status = "saved best checkpoint"
        else:
            checkpoint_status = "checkpoint unchanged"

        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {train_loss:.4f}, "
            f"train token accuracy: {train_accuracy:.4f}, "
            f"valid loss: {valid_loss:.4f}, "
            f"valid token accuracy: {valid_accuracy:.4f}, "
            f"{checkpoint_status}"
        )

    return history
