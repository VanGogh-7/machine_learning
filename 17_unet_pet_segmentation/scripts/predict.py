import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_datasets
from src.model import UNet
from src.utils import get_device, load_checkpoint
from src.visualize import save_segmentation_comparison


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    _, _, test_dataset = build_datasets(
        data_root=config.data_root,
        valid_ratio=config.valid_ratio,
        seed=config.seed,
        image_size=config.image_size,
        use_augmentation=config.use_augmentation,
    )
    model = UNet(
        in_channels=3,
        num_classes=config.n_classes,
        base_channels=config.base_channels,
        dropout=config.dropout,
    ).to(device)
    load_checkpoint(model, PROJECT_ROOT / config.model_path, device)
    model.eval()

    output_dir = PROJECT_ROOT / config.output_dir / "predictions"
    for index in range(5):
        image, true_mask = test_dataset[index]
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu()
        output_path = output_dir / f"sample_{index}.png"
        save_segmentation_comparison(image, true_mask, pred_mask, output_path)
        print(f"Saved prediction visualization: {output_path}")


if __name__ == "__main__":
    main()
