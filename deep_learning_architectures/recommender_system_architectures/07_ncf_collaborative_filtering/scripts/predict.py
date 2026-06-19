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
        print("Processed NCF feature metadata not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return
    if not config.processed_test_path.is_file():
        print("Processed NCF test CSV not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return
    if not checkpoint_path.is_file():
        print("Saved NCF checkpoint not found.")
        print("Run:")
        print("python scripts/train.py")
        return

    import pandas as pd
    import torch

    from src.model import NeuMF
    from src.utils import get_device, load_checkpoint, load_json

    metadata = load_json(config.processed_feature_meta_path)
    device = get_device()
    model = NeuMF(
        num_users=metadata["num_users"],
        num_items=metadata["num_items"],
        gmf_embedding_dim=config.user_embedding_dim,
        mlp_embedding_dim=config.mlp_embedding_dim,
        mlp_hidden_dims=config.mlp_hidden_dims,
        dropout=config.dropout,
    ).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    examples = pd.read_csv(config.processed_test_path).head(8)
    user_ids = torch.tensor(examples["user_id"].to_numpy(), dtype=torch.long).to(device)
    item_ids = torch.tensor(examples["item_id"].to_numpy(), dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(user_ids, item_ids)
        probabilities = torch.sigmoid(logits).cpu().tolist()

    item_id_to_title = metadata.get("item_id_to_title", {})
    for row, probability in zip(examples.itertuples(index=False), probabilities):
        item_id = int(getattr(row, "item_id"))
        predicted_class = int(probability >= 0.5)
        title = item_id_to_title.get(str(item_id))
        title_text = "" if title is None else f", title: {title}"
        print(
            f"user_id: {int(getattr(row, 'user_id'))}, "
            f"item_id: {item_id}{title_text}, "
            f"predicted interaction probability: {probability:.4f}, "
            f"predicted class: {predicted_class}, "
            f"true label: {float(getattr(row, 'label')):.0f}"
        )


if __name__ == "__main__":
    main()
