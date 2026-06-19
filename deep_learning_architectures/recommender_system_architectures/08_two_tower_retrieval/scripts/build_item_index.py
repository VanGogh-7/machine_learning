import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.config import TrainConfig
from src.model import TwoTowerRetrievalModel
from src.utils import get_device, load_checkpoint, load_json


def main() -> None:
    config = TrainConfig()
    output_dir = PROJECT_ROOT / config.output_dir
    checkpoint_path = output_dir / config.model_path
    item_ids_path = output_dir / config.item_index_path
    item_embeddings_path = output_dir / config.item_embedding_path

    if not config.processed_feature_meta_path.is_file():
        print("Processed Two-Tower feature metadata not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return
    if not checkpoint_path.is_file():
        print("Saved Two-Tower checkpoint not found.")
        print("Run:")
        print("python scripts/train.py")
        return

    metadata = load_json(config.processed_feature_meta_path)
    device = get_device()
    model = TwoTowerRetrievalModel(
        num_users=metadata["num_users"],
        num_items=metadata["num_items"],
        user_embedding_dim=config.user_embedding_dim,
        item_embedding_dim=config.item_embedding_dim,
        tower_hidden_dims=config.tower_hidden_dims,
        output_dim=config.output_dim,
        dropout=config.dropout,
        temperature=config.temperature,
    ).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    all_item_ids = torch.arange(metadata["num_items"], dtype=torch.long)
    item_vectors = []
    with torch.no_grad():
        for start in range(0, len(all_item_ids), config.batch_size):
            batch_item_ids = all_item_ids[start : start + config.batch_size].to(device)
            batch_vectors = model.encode_items(batch_item_ids)
            item_vectors.append(batch_vectors.cpu().numpy())

    item_embeddings = np.concatenate(item_vectors, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(item_ids_path, all_item_ids.numpy())
    np.save(item_embeddings_path, item_embeddings)

    print(f"Saved item ids: {item_ids_path}")
    print(f"Saved item embeddings: {item_embeddings_path}")
    print(f"Item ids shape: {tuple(all_item_ids.shape)}")
    print(f"Item embeddings shape: {tuple(item_embeddings.shape)}")


if __name__ == "__main__":
    main()
