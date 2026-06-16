import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import load_raw_splits, transform_dataframe
from src.model import XDeepFMCTRModel
from src.utils import get_device, load_checkpoint, load_json


def main() -> None:
    config = TrainConfig()
    output_dir = PROJECT_ROOT / config.output_dir
    checkpoint_path = output_dir / config.model_path
    feature_meta_path = output_dir / config.feature_meta_path

    device = get_device()
    print(f"Using device: {device}")

    metadata = load_json(feature_meta_path)
    model = XDeepFMCTRModel(
        num_numerical_features=len(metadata["numerical_feature_names"]),
        category_sizes=metadata["category_sizes"],
        embedding_dim=config.embedding_dim,
        cin_layer_sizes=config.cin_layer_sizes,
        cin_split_half=config.cin_split_half,
        deep_hidden_dims=config.deep_hidden_dims,
        dropout=config.dropout,
    ).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    if config.test_path.is_file():
        rows = pd.read_csv(config.test_path).head(5)
    else:
        _, _, test_df = load_raw_splits(
            config.train_path,
            config.valid_path,
            config.test_path,
            config.valid_ratio,
            config.test_ratio,
            config.seed,
        )
        rows = test_df.head(5)

    label_col = metadata["label_col"]
    has_labels = label_col in rows.columns
    numerical, categorical, labels = transform_dataframe(
        rows,
        metadata,
        require_label=has_labels,
    )
    numerical_tensor = torch.tensor(numerical, dtype=torch.float32, device=device)
    categorical_tensor = torch.tensor(categorical, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(numerical_tensor, categorical_tensor)
        probabilities = torch.sigmoid(logits)

    for index, (_, row) in enumerate(rows.iterrows()):
        probability = probabilities[index].item()
        predicted_class = int(probability >= 0.5)
        print(f"Row: {row.to_dict()}")
        print(f"Predicted click probability: {probability:.4f}")
        print(f"Predicted class: {predicted_class}")
        if labels is not None:
            print(f"True label: {int(labels[index])}")


if __name__ == "__main__":
    main()
