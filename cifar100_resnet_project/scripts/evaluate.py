import sys
from pathlib import Path

from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.engine import evaluate
from src.model import SmallResNet
from src.utils import get_device, load_checkpoint


def main() -> None:
    config = TrainConfig()
    device = get_device()
    _, _, test_loader = build_dataloaders(
        config.data_root, config.batch_size, config.num_workers, config.seed
    )
    model = SmallResNet(config.in_channels, config.n_classes).to(device)
    load_checkpoint(model, config.model_path, device)

    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Test top-1 accuracy: {metrics['top1']:.2%}")
    print(f"Test top-5 accuracy: {metrics['top5']:.2%}")


if __name__ == "__main__":
    main()
