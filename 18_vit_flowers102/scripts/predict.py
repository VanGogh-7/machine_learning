import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_datasets
from src.model import VisionTransformer
from src.utils import get_device, load_checkpoint
from src.visualize import save_prediction_grid


def label_name(label_index: int) -> str:
    return f"class_{label_index + 1}"


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    _, _, test_dataset = build_datasets(
        data_root=config.data_root,
        image_size=config.image_size,
        use_augmentation=config.use_augmentation,
    )
    model = VisionTransformer(
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.n_classes,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        mlp_dim=config.mlp_dim,
        num_encoder_layers=config.num_encoder_layers,
        dropout=config.dropout,
    ).to(device)
    load_checkpoint(model, PROJECT_ROOT / config.model_path, device)
    model.eval()

    images_for_grid = []
    titles = []
    for index in range(5):
        image, true_label = test_dataset[index]
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_label].item()
        print(f"Sample index: {index}")
        print(f"True label index: {true_label}")
        print(f"True label name: {label_name(true_label)}")
        print(f"Predicted label index: {predicted_label}")
        print(f"Predicted label name: {label_name(predicted_label)}")
        print(f"Confidence: {confidence:.4f}")
        images_for_grid.append(image)
        titles.append(
            f"true: {label_name(true_label)}\n"
            f"pred: {label_name(predicted_label)} ({confidence:.2f})"
        )

    output_path = PROJECT_ROOT / config.output_dir / "prediction_grid.png"
    save_prediction_grid(images_for_grid, titles, output_path)
    print(f"Saved prediction grid to {output_path}")


if __name__ == "__main__":
    main()

