from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils import save_checkpoint, save_history


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
    tgt_pad_idx: int,
    clip_grad_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src_input_ids, target_ids in data_loader:
        src_input_ids = src_input_ids.to(device)
        target_ids = target_ids.to(device)
        decoder_input = target_ids[:, :-1]
        decoder_target = target_ids[:, 1:]

        optimizer.zero_grad()
        logits = model(src_input_ids, decoder_input)
        loss = compute_loss(logits, decoder_target, loss_fn)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        token_count = (decoder_target != tgt_pad_idx).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    return total_loss / total_tokens


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    tgt_pad_idx: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src_input_ids, target_ids in data_loader:
            src_input_ids = src_input_ids.to(device)
            target_ids = target_ids.to(device)
            decoder_input = target_ids[:, :-1]
            decoder_target = target_ids[:, 1:]
            logits = model(src_input_ids, decoder_input)
            loss = compute_loss(logits, decoder_target, loss_fn)

            token_count = (decoder_target != tgt_pad_idx).sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count

    return total_loss / total_tokens


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    tgt_pad_idx: int,
    clip_grad_norm: float,
    n_epochs: int,
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, list[float]]:
    history = {"train_losses": [], "valid_losses": []}
    best_valid_loss = float("inf")

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            tgt_pad_idx=tgt_pad_idx,
            clip_grad_norm=clip_grad_norm,
        )
        valid_loss = evaluate(
            model=model,
            data_loader=valid_loader,
            loss_fn=loss_fn,
            device=device,
            tgt_pad_idx=tgt_pad_idx,
        )
        history["train_losses"].append(train_loss)
        history["valid_losses"].append(valid_loss)
        save_history(history, history_path)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model, checkpoint_path)
            checkpoint_status = "saved best checkpoint"
        else:
            checkpoint_status = "checkpoint unchanged"

        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {train_loss:.4f}, "
            f"valid loss: {valid_loss:.4f}, "
            f"{checkpoint_status}"
        )

    return history
