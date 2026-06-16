import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_datasets
from src.model import InceptionCIFAR100
from src.utils import get_device, load_checkpoint


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    _, _, test_dataset = build_datasets(
        data_root=config.data_root,
        valid_ratio=config.valid_ratio,
        seed=config.seed,
        use_augmentation=config.use_augmentation,
    )
    model = InceptionCIFAR100(num_classes=config.n_classes).to(device)
    load_checkpoint(model, PROJECT_ROOT / config.model_path, device)
    model.eval()

    for index in range(5):
        image, true_label = test_dataset[index]
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_label].item()
        print(f"Sample index: {index}")
        print(f"True label index: {true_label}")
        print(f"True label name: {test_dataset.classes[true_label]}")
        print(f"Predicted label index: {predicted_label}")
        print(f"Predicted label name: {test_dataset.classes[predicted_label]}")
        print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
