import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_datasets
from src.model import ResNetCIFAR10
from src.utils import get_device, load_checkpoint


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    _, _, test_dataset = build_datasets(
        data_root=config.data_root,
        valid_size=config.valid_size,
        seed=config.seed,
    )
    image, true_label = test_dataset[0]

    model = ResNetCIFAR10(
        in_channels=config.in_channels,
        n_classes=config.n_classes,
    ).to(device)
    checkpoint_path = PROJECT_ROOT / config.model_path
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        predicted_label = logits.argmax(dim=1).item()

    print(f"True label index: {true_label}")
    print(f"True class name: {test_dataset.classes[true_label]}")
    print(f"Predicted label index: {predicted_label}")
    print(f"Predicted class name: {test_dataset.classes[predicted_label]}")


if __name__ == "__main__":
    main()
