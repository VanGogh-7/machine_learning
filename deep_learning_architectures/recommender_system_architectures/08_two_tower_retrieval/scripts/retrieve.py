import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
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
    if not config.processed_test_path.is_file():
        print("Processed Two-Tower test CSV not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return
    if not checkpoint_path.is_file():
        print("Saved Two-Tower checkpoint not found.")
        print("Run:")
        print("python scripts/train.py")
        return
    if not item_ids_path.is_file() or not item_embeddings_path.is_file():
        print("Item index not found.")
        print("Run:")
        print("python scripts/build_item_index.py")
        return

    metadata = load_json(config.processed_feature_meta_path)
    item_ids = np.load(item_ids_path)
    item_embeddings = np.load(item_embeddings_path)
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

    examples = pd.read_csv(config.processed_test_path).head(5)
    item_embedding_tensor = torch.tensor(
        item_embeddings,
        dtype=torch.float32,
        device=device,
    )
    item_id_to_title = metadata.get("item_id_to_title", {})

    with torch.no_grad():
        for user_id in examples["user_id"].astype(int).tolist():
            user_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
            user_vector = model.encode_users(user_tensor)
            scores = (user_vector @ item_embedding_tensor.T).squeeze(0)
            effective_top_k = min(config.top_k, scores.numel())
            top_scores, top_positions = scores.topk(effective_top_k)
            top_item_ids = item_ids[top_positions.cpu().numpy()]

            print(f"encoded user id: {user_id}")
            print(f"top_{effective_top_k} recommended encoded item ids:")
            for item_id, score in zip(top_item_ids.tolist(), top_scores.cpu().tolist()):
                title = item_id_to_title.get(str(int(item_id)))
                title_text = "" if title is None else f", title: {title}"
                print(f"  item_id: {int(item_id)}, score: {score:.4f}{title_text}")


if __name__ == "__main__":
    main()
