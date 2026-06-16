import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig


def processed_dataset_exists(config: TrainConfig) -> bool:
    return all(
        path.is_file()
        for path in (
            config.processed_train_path,
            config.processed_valid_path,
            config.processed_test_path,
            config.processed_feature_meta_path,
        )
    )


def title_for_item(item_id: int, metadata: dict) -> str:
    if item_id == metadata["padding_item_id"]:
        return "<padding>"
    return metadata.get("item_id_to_title", {}).get(str(item_id), f"item {item_id}")


def main() -> None:
    config = TrainConfig()
    output_dir = PROJECT_ROOT / config.output_dir
    checkpoint_path = output_dir / config.model_path

    if not processed_dataset_exists(config):
        print("Processed DIEN dataset not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return
    if not checkpoint_path.is_file():
        print(f"Model checkpoint not found: {checkpoint_path}")
        print("Run:")
        print("python scripts/train.py")
        return

    import torch

    from src.data import ProcessedDIENDataset
    from src.model import DIENCTRModel
    from src.utils import get_device, load_checkpoint, load_json

    device = get_device()
    print(f"Using device: {device}")

    metadata = load_json(config.processed_feature_meta_path)
    model = DIENCTRModel(
        num_items=metadata["num_items"],
        embedding_dim=config.embedding_dim,
        gru_hidden_dim=config.gru_hidden_dim,
        max_history_length=metadata["max_history_length"],
        mlp_hidden_dims=config.mlp_hidden_dims,
        dropout=config.dropout,
    ).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    test_dataset = ProcessedDIENDataset(
        config.processed_test_path,
        max_history_length=int(metadata["max_history_length"]),
    )
    sample_count = min(5, len(test_dataset))
    batch = [test_dataset[index] for index in range(sample_count)]
    target_items = torch.stack([sample[0] for sample in batch])
    histories = torch.stack([sample[1] for sample in batch])
    next_histories = torch.stack([sample[2] for sample in batch])
    history_masks = torch.stack([sample[3] for sample in batch])
    aux_masks = torch.stack([sample[4] for sample in batch])
    labels = torch.stack([sample[5] for sample in batch])

    with torch.no_grad():
        outputs = model(
            target_item_ids=target_items.to(device),
            history_item_ids=histories.to(device),
            next_history_item_ids=next_histories.to(device),
            history_mask=history_masks.to(device),
            aux_mask=aux_masks.to(device),
        )
        probabilities = torch.sigmoid(outputs["logits"])

    for index in range(sample_count):
        target_item_id = int(target_items[index].item())
        history_ids = [
            int(item_id)
            for item_id, mask_value in zip(
                histories[index].tolist(),
                history_masks[index].tolist(),
            )
            if mask_value > 0
        ]
        probability = probabilities[index].item()
        predicted_class = int(probability >= 0.5)

        print(f"Target item id: {target_item_id}")
        print(f"Target movie title: {title_for_item(target_item_id, metadata)}")
        print(f"History item ids: {history_ids}")
        print(f"Predicted click probability: {probability:.4f}")
        print(f"Predicted class: {predicted_class}")
        print(f"True label: {int(labels[index].item())}")


if __name__ == "__main__":
    main()
