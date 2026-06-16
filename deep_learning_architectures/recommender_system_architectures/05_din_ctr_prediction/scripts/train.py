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
    history_path = output_dir / config.history_path

    if not processed_dataset_exists(config):
        print("Processed DIN dataset not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return

    import torch
    from torch import nn

    from src.data import build_dataloaders
    from src.engine import evaluate, train
    from src.model import DINCTRModel
    from src.utils import clear_memory, get_device, load_checkpoint, set_seed

    clear_memory()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, metadata = build_dataloaders(
        config=config,
        device=device,
    )

    print(f"Processed train path: {config.processed_train_path}")
    print(f"Processed validation path: {config.processed_valid_path}")
    print(f"Processed test path: {config.processed_test_path}")
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(valid_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")
    print(f"Number of items: {metadata['num_items']}")
    print(f"Padding item id: {metadata['padding_item_id']}")
    print(f"Max history length: {metadata['max_history_length']}")
    print(f"Positive rating threshold: {metadata['positive_rating_threshold']}")

    model = DINCTRModel(
        num_items=metadata["num_items"],
        embedding_dim=config.embedding_dim,
        max_history_length=metadata["max_history_length"],
        activation_hidden_dims=config.activation_hidden_dims,
        mlp_hidden_dims=config.mlp_hidden_dims,
        dropout=config.dropout,
    ).to(device)

    target_items, histories, masks, labels = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        outputs = model(
            target_items.to(device),
            histories.to(device),
            masks.to(device),
        )
    print(f"Target item ids shape: {tuple(target_items.shape)}")
    print(f"History item ids shape: {tuple(histories.shape)}")
    print(f"History mask shape: {tuple(masks.shape)}")
    print(f"Labels shape: {tuple(labels.shape)}")
    print(f"Logits shape: {tuple(outputs.shape)}")

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
    test_loss, test_accuracy, test_auc = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
    )
    test_auc_text = "n/a" if test_auc is None else f"{test_auc:.4f}"
    print(f"Best model saved to {checkpoint_path}")
    print(f"Feature metadata loaded from {config.processed_feature_meta_path}")
    print(f"Training history saved to {history_path}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Final test AUC: {test_auc_text}")


if __name__ == "__main__":
    main()
