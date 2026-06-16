import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import load_raw_splits, transform_dataframe
from src.model import DeepFMCTRModel
from src.utils import get_device, load_checkpoint, load_json


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    metadata = load_json(PROJECT_ROOT / config.feature_meta_path)
    model = DeepFMCTRModel(
        num_numerical_features=len(metadata["numerical_feature_names"]),
        category_sizes=metadata["category_sizes"],
        embedding_dim=config.embedding_dim,
        deep_hidden_dims=config.deep_hidden_dims,
        dropout=config.dropout,
    ).to(device)
    load_checkpoint(model, PROJECT_ROOT / config.model_path, device)
    model.eval()

    if config.test_path.is_file():
        test_rows = pd.read_csv(config.test_path).head(5)
    else:
        _, _, test_rows = load_raw_splits(
            config.train_path,
            config.valid_path,
            config.test_path,
            config.valid_ratio,
            config.test_ratio,
            config.seed,
        )
        test_rows = test_rows.head(5)
    numerical, categorical, labels = transform_dataframe(test_rows, metadata)
    numerical_tensor = torch.tensor(numerical, dtype=torch.float32, device=device)
    categorical_tensor = torch.tensor(categorical, dtype=torch.long, device=device)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(numerical_tensor, categorical_tensor))

    for index, (_, row) in enumerate(test_rows.iterrows()):
        probability = probabilities[index].item()
        predicted_class = int(probability >= 0.5)
        print(f"Row: {row.to_dict()}")
        print(f"Predicted click probability: {probability:.4f}")
        print(f"Predicted class: {predicted_class}")
        print(f"True label: {int(labels[index])}")


if __name__ == "__main__":
    main()
