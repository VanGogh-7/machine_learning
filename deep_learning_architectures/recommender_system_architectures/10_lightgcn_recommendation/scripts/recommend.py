import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig


def main() -> None:
    config = TrainConfig()
    output_dir = PROJECT_ROOT / config.output_dir
    checkpoint_path = output_dir / config.model_path

    if not config.processed_feature_meta_path.is_file():
        print("Processed LightGCN feature metadata not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return
    if not config.processed_train_path.is_file() or not config.processed_test_path.is_file():
        print("Processed LightGCN CSV files were not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return
    if not checkpoint_path.is_file():
        print("Saved LightGCN checkpoint not found.")
        print("Run:")
        print("python scripts/train.py")
        return

    import pandas as pd
    import torch

    from src.data import (
        build_normalized_adj,
        build_user_positive_items,
        read_positive_edges,
    )
    from src.model import LightGCN
    from src.utils import get_device, load_checkpoint, load_json

    metadata = load_json(config.processed_feature_meta_path)
    train_edges = read_positive_edges(config.processed_train_path)
    test_edges = read_positive_edges(config.processed_test_path)
    train_user_positive_items = build_user_positive_items(train_edges)
    normalized_adj = build_normalized_adj(
        train_edges=train_edges,
        num_users=metadata["num_users"],
        num_items=metadata["num_items"],
    )

    device = get_device()
    normalized_adj = normalized_adj.to(device)
    model = LightGCN(
        num_users=metadata["num_users"],
        num_items=metadata["num_items"],
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
    ).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    user_ids = test_edges["user_id"].drop_duplicates().astype(int).head(5).tolist()
    item_id_to_title = metadata.get("item_id_to_title", {})

    with torch.no_grad():
        for user_id in user_ids:
            user_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
            scores = model.full_sort_scores(user_tensor, normalized_adj).squeeze(0)
            seen_items = train_user_positive_items.get(user_id, set())
            if seen_items:
                scores[list(seen_items)] = -float("inf")
            effective_top_k = min(config.top_k, scores.numel())
            top_scores, top_item_ids = scores.topk(effective_top_k)

            print(f"encoded user id: {user_id}")
            print(f"top_{effective_top_k} recommended encoded item ids:")
            for item_id, score in zip(top_item_ids.cpu().tolist(), top_scores.cpu().tolist()):
                title = item_id_to_title.get(str(int(item_id)))
                title_text = "" if title is None else f", title: {title}"
                print(f"  item_id: {int(item_id)}, score: {score:.4f}{title_text}")


if __name__ == "__main__":
    main()
