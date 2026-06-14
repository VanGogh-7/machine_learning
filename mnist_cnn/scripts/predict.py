import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.model import MNISTCNN
from src.utils import get_device


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    _, valid_loader, _ = build_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    model = MNISTCNN(
        in_channels=config.in_channels,
        n_classes=config.n_classes,
    ).to(device)

    state_dict = torch.load(config.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    inputs, true_labels = next(iter(valid_loader))
    inputs = inputs[:5].to(device)
    true_labels = true_labels[:5]

    with torch.no_grad():
        predicted_labels = model(inputs).argmax(dim=1).cpu()

    print("Predicted labels:", predicted_labels.tolist())
    print("True labels:     ", true_labels.tolist())


if __name__ == "__main__":
    main()
