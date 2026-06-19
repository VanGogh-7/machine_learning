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
        print("Processed Two-Tower dataset not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return

    import torch
    from torch import nn

    from src.data import build_dataloaders
    from src.engine import evaluate, train
    from src.model import TwoTowerRetrievalModel
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
    print(f"Number of users: {metadata['num_users']}")
    print(f"Number of items: {metadata['num_items']}")
    print(f"User embedding dimension: {config.user_embedding_dim}")
    print(f"Item embedding dimension: {config.item_embedding_dim}")
    print(f"Output dimension: {config.output_dim}")
    print(f"Temperature: {config.temperature}")

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

    user_ids, item_ids = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        outputs = model(user_ids.to(device), item_ids.to(device))
    print(f"user_ids shape: {tuple(user_ids.shape)}")
    print(f"item_ids shape: {tuple(item_ids.shape)}")
    print(f"user_vectors shape: {tuple(outputs['user_vectors'].shape)}")
    print(f"item_vectors shape: {tuple(outputs['item_vectors'].shape)}")
    print(f"scores shape: {tuple(outputs['scores'].shape)}")

    loss_fn = nn.CrossEntropyLoss()
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
    print(f"Best model saved to {checkpoint_path}")
    print(f"Feature metadata saved to {feature_meta_path}")
    print(f"Processed feature metadata loaded from {config.processed_feature_meta_path}")
    print(f"Training history saved to {history_path}")
    print(f"Final test loss: {test_metrics['loss']:.4f}")
    print(
        f"Final test retrieval accuracy: "
        f"{test_metrics['retrieval_accuracy']:.4f}"
    )
    print(f"Final test Recall@1: {test_metrics['recall_at_1']:.4f}")
    print(f"Final test Recall@5: {test_metrics['recall_at_5']:.4f}")
    print(f"Final test Recall@10: {test_metrics['recall_at_10']:.4f}")


if __name__ == "__main__":
    main()
