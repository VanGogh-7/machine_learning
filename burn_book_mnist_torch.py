import os
import time
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parent
MNIST_ROOT = REPO_ROOT / "datasets" / "mnist"


@dataclass
class ModelConfig:
    num_classes: int = 10
    hidden_size: int = 512
    dropout: float = 0.5


@dataclass
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42
    learning_rate: float = 1.0e-4
    artifact_dir: str = "/tmp/guide_torch"


class BurnBookMnistModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=True,
        )

        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.dropout = nn.Dropout(p=config.dropout)
        self.linear1 = nn.Linear(16 * 8 * 8, config.hidden_size, bias=True)
        self.linear2 = nn.Linear(config.hidden_size, config.num_classes, bias=True)
        self.activation = nn.ReLU()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Burn's input shape is [batch_size, height, width].
        # PyTorch Conv2d needs [batch_size, channels, height, width].
        if images.ndim == 3:
            x = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2])
        elif images.ndim == 4:
            x = images
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {tuple(images.shape)}")

        x = self.conv1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.pool(x)

        x = x.reshape(x.shape[0], 16 * 8 * 8)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)

        return self.linear2(x)


def set_seed(seed: int) -> None:
    # Set seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync_if_cuda(device: torch.device) -> None:
    # CUDA operations are asynchronous, so synchronize before timing.
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dataloaders(config: TrainingConfig):
    # Match Burn's preprocessing: image / 255, then normalize with mean=0.1307 and std=0.3081.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts uint8 [0, 255] to float [0, 1], shape [1, 28, 28].
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.squeeze(0)),  # Match Burn shape: [28, 28].
        ]
    )

    train_dataset = datasets.MNIST(
        root=MNIST_ROOT,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=MNIST_ROOT,
        train=False,
        download=True,
        transform=transform,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    return train_loader, test_loader, test_dataset


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
    }


def train(config: TrainingConfig):
    set_seed(config.seed)

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(config.artifact_dir, exist_ok=True)

    train_loader, test_loader, test_dataset = get_dataloaders(config)

    model_config = ModelConfig(num_classes=10, hidden_size=512, dropout=0.5)
    model = BurnBookMnistModel(model_config).to(device)

    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    sync_if_cuda(device)
    train_start = time.perf_counter()

    for epoch in range(1, config.num_epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_count = 0

        sync_if_cuda(device)
        epoch_start = time.perf_counter()

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            batch_size = targets.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_correct += (logits.argmax(dim=1) == targets).sum().item()
            epoch_count += batch_size

        sync_if_cuda(device)
        epoch_time = time.perf_counter() - epoch_start

        train_metrics = {
            "loss": epoch_loss / epoch_count,
            "accuracy": epoch_correct / epoch_count,
        }

        test_metrics = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:02d}/{config.num_epochs} | "
            f"time={epoch_time:.3f}s | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_acc={test_metrics['accuracy']:.4f}"
        )

    sync_if_cuda(device)
    train_time = time.perf_counter() - train_start

    model_path = os.path.join(config.artifact_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config.__dict__,
            "training_config": config.__dict__,
        },
        model_path,
    )

    print(f"Training time: {train_time:.3f}s")
    print(f"Saved model to: {model_path}")

    return model_path, test_dataset


def infer_single(model_path: str, item_index: int = 42) -> None:
    device = get_device()

    checkpoint = torch.load(model_path, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])

    model = BurnBookMnistModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.squeeze(0)),
        ]
    )

    test_dataset = datasets.MNIST(
        root=MNIST_ROOT,
        train=False,
        download=True,
        transform=transform,
    )

    image, label = test_dataset[item_index]
    image = image.unsqueeze(0).to(device)

    sync_if_cuda(device)
    start = time.perf_counter()

    with torch.no_grad():
        logits = model(image)
        predicted = logits.argmax(dim=1).item()

    sync_if_cuda(device)
    elapsed = time.perf_counter() - start

    print(f"Single inference item index: {item_index}")
    print(f"Predicted: {predicted}, Expected: {label}")
    print(f"Single inference time: {elapsed * 1000:.4f} ms")


def benchmark_inference(model_path: str, batch_size: int = 64, warmup_steps: int = 20, measure_steps: int = 100):
    device = get_device()

    checkpoint = torch.load(model_path, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])

    model = BurnBookMnistModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(batch_size, 28, 28, device=device)

    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(dummy)

        sync_if_cuda(device)
        start = time.perf_counter()

        for _ in range(measure_steps):
            _ = model(dummy)

        sync_if_cuda(device)
        elapsed = time.perf_counter() - start

    avg_batch_ms = elapsed / measure_steps * 1000
    avg_item_ms = avg_batch_ms / batch_size

    print(f"Inference benchmark:")
    print(f"  batch_size: {batch_size}")
    print(f"  measure_steps: {measure_steps}")
    print(f"  avg_batch_time: {avg_batch_ms:.4f} ms")
    print(f"  avg_item_time: {avg_item_ms:.6f} ms")


def main():
    config = TrainingConfig(
        num_epochs=10,
        batch_size=64,
        num_workers=4,
        seed=42,
        learning_rate=1.0e-4,
        artifact_dir="/tmp/guide_torch",
    )

    model_path, _ = train(config)
    infer_single(model_path, item_index=42)
    benchmark_inference(model_path, batch_size=64)


if __name__ == "__main__":
    main()
