from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.utils import save_checkpoint, save_json


def vae_loss_function(
    reconstructed_images: torch.Tensor,
    images: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = F.binary_cross_entropy(
        reconstructed_images,
        images,
        reduction="sum",
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
) -> tuple[float, float, float]:
    model.train()
    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    kl_loss_sum = 0.0
    total_samples = 0

    for images, _ in data_loader:
        images = images.to(device)

        optimizer.zero_grad()
        reconstructed_images, mu, logvar = model(images)
        total_loss, reconstruction_loss, kl_loss = vae_loss_function(
            reconstructed_images,
            images,
            mu,
            logvar,
            beta,
        )
        total_loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss_sum += total_loss.item()
        reconstruction_loss_sum += reconstruction_loss.item()
        kl_loss_sum += kl_loss.item()
        total_samples += batch_size

    return (
        total_loss_sum / total_samples,
        reconstruction_loss_sum / total_samples,
        kl_loss_sum / total_samples,
    )


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    beta: float,
) -> tuple[float, float, float]:
    model.eval()
    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    kl_loss_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            reconstructed_images, mu, logvar = model(images)
            total_loss, reconstruction_loss, kl_loss = vae_loss_function(
                reconstructed_images,
                images,
                mu,
                logvar,
                beta,
            )

            batch_size = images.size(0)
            total_loss_sum += total_loss.item()
            reconstruction_loss_sum += reconstruction_loss.item()
            kl_loss_sum += kl_loss.item()
            total_samples += batch_size

    return (
        total_loss_sum / total_samples,
        reconstruction_loss_sum / total_samples,
        kl_loss_sum / total_samples,
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    n_epochs: int,
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, list[float]]:
    history = {
        "train_total_losses": [],
        "train_reconstruction_losses": [],
        "train_kl_losses": [],
        "valid_total_losses": [],
        "valid_reconstruction_losses": [],
        "valid_kl_losses": [],
    }
    best_valid_loss = float("inf")

    for epoch in range(n_epochs):
        train_total, train_reconstruction, train_kl = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=beta,
        )
        valid_total, valid_reconstruction, valid_kl = evaluate(
            model=model,
            data_loader=valid_loader,
            device=device,
            beta=beta,
        )

        history["train_total_losses"].append(train_total)
        history["train_reconstruction_losses"].append(train_reconstruction)
        history["train_kl_losses"].append(train_kl)
        history["valid_total_losses"].append(valid_total)
        history["valid_reconstruction_losses"].append(valid_reconstruction)
        history["valid_kl_losses"].append(valid_kl)
        save_json(history, history_path)

        if valid_total < best_valid_loss:
            best_valid_loss = valid_total
            save_checkpoint(model, checkpoint_path)
            checkpoint_status = "saved best checkpoint"
        else:
            checkpoint_status = "checkpoint unchanged"

        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train total loss: {train_total:.4f}, "
            f"train reconstruction loss: {train_reconstruction:.4f}, "
            f"train KL loss: {train_kl:.4f}, "
            f"valid total loss: {valid_total:.4f}, "
            f"valid reconstruction loss: {valid_reconstruction:.4f}, "
            f"valid KL loss: {valid_kl:.4f}, "
            f"{checkpoint_status}"
        )

    return history

