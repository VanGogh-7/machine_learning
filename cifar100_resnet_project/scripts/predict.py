import sys
from pathlib import Path

import torch
from torchvision.datasets import CIFAR100

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_eval_transform
from src.model import SmallResNet
from src.utils import get_device, load_checkpoint


def main() -> None:
    config = TrainConfig()
    device = get_device()
    dataset = CIFAR100(
        root=config.data_root, train=False, download=True, transform=build_eval_transform()
    )
    model = SmallResNet(config.in_channels, config.n_classes).to(device)
    load_checkpoint(model, config.model_path, device)
    model.eval()

    with torch.no_grad():
        for index in range(5):
            image, true_index = dataset[index]
            logits = model(image.unsqueeze(0).to(device))
            predicted_index = logits.argmax(dim=1).item()
            print(
                f"Image {index}: predicted {predicted_index} ({dataset.classes[predicted_index]}), "
                f"true {true_index} ({dataset.classes[true_index]})"
            )


if __name__ == "__main__":
    main()
