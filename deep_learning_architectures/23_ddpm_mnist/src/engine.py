from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.diffusion import GaussianDiffusion
from src.utils import save_checkpoint, save_json
from src.visualize import save_image_grid


def train_one_epoch(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, _ in data_loader:
        images = images.to(device)
        batch_size = images.size(0)
        timesteps = torch.randint(
            0,
            diffusion.num_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        noisy_images, noise = diffusion.q_sample(images, timesteps)

        optimizer.zero_grad()
        predicted_noise = model(noisy_images, timesteps)
        loss = loss_fn(predicted_noise, noise)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            batch_size = images.size(0)
            timesteps = torch.randint(
                0,
                diffusion.num_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            noisy_images, noise = diffusion.q_sample(images, timesteps)
            predicted_noise = model(noisy_images, timesteps)
            loss = loss_fn(predicted_noise, noise)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def train(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    n_epochs: int,
    checkpoint_path: Path,
    history_path: Path,
    output_dir: Path,
    sample_interval: int,
    num_generation_samples: int,
    image_channels: int,
    image_size: int,
) -> dict[str, list[float]]:
    history = {
        "train_losses": [],
        "valid_losses": [],
    }
    best_valid_loss = float("inf")

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(
            model=model,
            diffusion=diffusion,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        valid_loss = evaluate(
            model=model,
            diffusion=diffusion,
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

        if (epoch + 1) % sample_interval == 0 or epoch == 0:
            samples = diffusion.sample(
                model=model,
                shape=(num_generation_samples, image_channels, image_size, image_size),
                device=device,
            )
            save_image_grid(samples, output_dir / f"samples_epoch_{epoch + 1:03d}.png")

        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {train_loss:.6f}, "
            f"valid loss: {valid_loss:.6f}, "
            f"{checkpoint_status}"
        )

    return history

