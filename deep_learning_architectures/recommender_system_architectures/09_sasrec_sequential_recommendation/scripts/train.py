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


def main() -> None:
    config = TrainConfig()
    output_dir = PROJECT_ROOT / config.output_dir
    checkpoint_path = output_dir / config.model_path
    feature_meta_path = output_dir / config.feature_meta_path
    history_path = output_dir / config.history_path

    if not processed_dataset_exists(config):
        print("Processed SASRec dataset not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return

    import torch
    from torch import nn

    from src.data import build_dataloaders
    from src.engine import evaluate, train
    from src.model import SASRecModel
    from src.utils import get_device, load_checkpoint, save_json, set_seed

    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, metadata = build_dataloaders(
        config=config,
        device=device,
    )
    save_json(metadata, feature_meta_path)

    print(f"Processed train path: {config.processed_train_path}")
    print(f"Processed validation path: {config.processed_valid_path}")
    print(f"Processed test path: {config.processed_test_path}")
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(valid_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")
    print(f"Number of items: {metadata['num_items']}")
    print(f"Padding item id: {metadata['padding_item_id']}")
    print(f"Max sequence length: {metadata['max_sequence_length']}")
    print(f"Embedding dimension: {config.embedding_dim}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    print(f"Number of Transformer blocks: {config.num_transformer_blocks}")

    model = SASRecModel(
        num_items=metadata["num_items"],
        max_sequence_length=metadata["max_sequence_length"],
        embedding_dim=config.embedding_dim,
        num_attention_heads=config.num_attention_heads,
        num_transformer_blocks=config.num_transformer_blocks,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
    ).to(device)

    input_item_ids, target_item_ids, labels = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        logits = model(input_item_ids.to(device), target_item_ids.to(device))
    print(f"input_item_ids shape: {tuple(input_item_ids.shape)}")
    print(f"target_item_ids shape: {tuple(target_item_ids.shape)}")
    print(f"labels shape: {tuple(labels.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        n_epochs=config.n_epochs,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    load_checkpoint(model, checkpoint_path, device)
    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )
    test_auc_text = "n/a" if test_metrics["auc"] is None else f"{test_metrics['auc']:.4f}"
    print(f"Best model saved to {checkpoint_path}")
    print(f"Feature metadata saved to {feature_meta_path}")
    print(f"Processed feature metadata loaded from {config.processed_feature_meta_path}")
    print(f"Training history saved to {history_path}")
    print(f"Final test loss: {test_metrics['loss']:.4f}")
    print(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final test AUC: {test_auc_text}")


if __name__ == "__main__":
    main()
