import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_datasets
from src.model import AlexNetCIFAR10
from src.utils import get_device, load_checkpoint


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    _, test_data = build_datasets(config.data_root)
    image, true_label = test_data[0]

    model = AlexNetCIFAR10(
        in_channels=config.in_channels,
        n_classes=config.n_classes,
    ).to(device)
    checkpoint_path = PROJECT_ROOT / config.model_path
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        predicted_label = logits.argmax(dim=1).item()

    print(f"True label: {true_label} ({test_data.classes[true_label]})")
    print(f"Predicted label: {predicted_label} ({test_data.classes[predicted_label]})")


if __name__ == "__main__":
    main()
