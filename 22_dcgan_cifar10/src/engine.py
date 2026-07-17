from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils import save_checkpoint, save_json
from src.visualize import save_generated_grid


def discriminator_accuracy(logits: torch.Tensor, target_is_real: bool) -> int:
    predictions = torch.sigmoid(logits) >= 0.5
    targets = torch.ones_like(predictions) if target_is_real else torch.zeros_like(predictions)
    return (predictions == targets).sum().item()


def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    data_loader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    latent_dim: int,
) -> tuple[float, float, float, float]:
    generator.train()
    discriminator.train()
    total_d_loss = 0.0
    total_g_loss = 0.0
    total_real_correct = 0
    total_fake_correct = 0
    total_samples = 0

    for real_images, _ in data_loader:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        # Update only the discriminator using real images and detached fakes.
        optimizer_d.zero_grad()
        real_logits = discriminator(real_images)
        real_loss = loss_fn(real_logits, real_labels)

        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(z)
        fake_logits = discriminator(fake_images.detach())
        fake_loss = loss_fn(fake_logits, fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()

        # Update only the generator so fake images are classified as real.
        optimizer_g.zero_grad()
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(z)
        generator_logits = discriminator(fake_images)
        g_loss = loss_fn(generator_logits, real_labels)
        g_loss.backward()
        optimizer_g.step()

        total_d_loss += d_loss.item() * batch_size
        total_g_loss += g_loss.item() * batch_size
        total_real_correct += discriminator_accuracy(real_logits.detach(), target_is_real=True)
        total_fake_correct += discriminator_accuracy(fake_logits.detach(), target_is_real=False)
        total_samples += batch_size

    return (
        total_d_loss / total_samples,
        total_g_loss / total_samples,
        total_real_correct / total_samples,
        total_fake_correct / total_samples,
    )


def train(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    latent_dim: int,
    n_epochs: int,
    generator_path: Path,
    discriminator_path: Path,
    history_path: Path,
    output_dir: Path,
    fixed_noise: torch.Tensor,
    sample_interval: int,
) -> dict[str, list[float]]:
    history = {
        "discriminator_losses": [],
        "generator_losses": [],
        "real_accuracies": [],
        "fake_accuracies": [],
    }

    for epoch in range(n_epochs):
        d_loss, g_loss, real_accuracy, fake_accuracy = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            data_loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            loss_fn=loss_fn,
            device=device,
            latent_dim=latent_dim,
        )

        history["discriminator_losses"].append(d_loss)
        history["generator_losses"].append(g_loss)
        history["real_accuracies"].append(real_accuracy)
        history["fake_accuracies"].append(fake_accuracy)

        save_checkpoint(generator, generator_path)
        save_checkpoint(discriminator, discriminator_path)
        save_json(history, history_path)

        if (epoch + 1) % sample_interval == 0 or epoch == 0:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise)
            save_generated_grid(samples, output_dir / f"samples_epoch_{epoch + 1:03d}.png")

        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"discriminator loss: {d_loss:.4f}, "
            f"generator loss: {g_loss:.4f}, "
            f"real accuracy: {real_accuracy:.4f}, "
            f"fake accuracy: {fake_accuracy:.4f}, "
            "saved latest checkpoints"
        )

    return history

