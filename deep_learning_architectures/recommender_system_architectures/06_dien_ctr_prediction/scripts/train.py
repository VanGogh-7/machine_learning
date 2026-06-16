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
        print("Processed DIEN dataset not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return

    import torch
    from torch import nn

    from src.data import build_dataloaders
    from src.engine import evaluate, train
    from src.model import DIENCTRModel
    from src.utils import (
        clear_memory,
        get_device,
        load_checkpoint,
        load_json,
        save_json,
        set_seed,
    )

    clear_memory()
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
    print(f"Max history length: {metadata['max_history_length']}")
    print(f"Embedding dimension: {config.embedding_dim}")
    print(f"GRU hidden dimension: {config.gru_hidden_dim}")
    print(f"Auxiliary loss weight: {config.aux_loss_weight}")

    model = DIENCTRModel(
        num_items=metadata["num_items"],
        embedding_dim=config.embedding_dim,
        gru_hidden_dim=config.gru_hidden_dim,
        max_history_length=metadata["max_history_length"],
        mlp_hidden_dims=config.mlp_hidden_dims,
        dropout=config.dropout,
    ).to(device)

    (
        target_items,
        histories,
        next_histories,
        history_masks,
        aux_masks,
        labels,
    ) = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        outputs = model(
            target_item_ids=target_items.to(device),
            history_item_ids=histories.to(device),
            next_history_item_ids=next_histories.to(device),
            history_mask=history_masks.to(device),
            aux_mask=aux_masks.to(device),
        )
    print(f"Target item ids shape: {tuple(target_items.shape)}")
    print(f"History item ids shape: {tuple(histories.shape)}")
    print(f"Next history item ids shape: {tuple(next_histories.shape)}")
    print(f"History mask shape: {tuple(history_masks.shape)}")
    print(f"Aux mask shape: {tuple(aux_masks.shape)}")
    print(f"Labels shape: {tuple(labels.shape)}")
    print(f"Model output logits shape: {tuple(outputs['logits'].shape)}")
    print(f"Model aux_logits shape: {tuple(outputs['aux_logits'].shape)}")
    print(f"Model attention_weights shape: {tuple(outputs['attention_weights'].shape)}")

    ctr_loss_fn = nn.BCEWithLogitsLoss()
    aux_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
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
        ctr_loss_fn=ctr_loss_fn,
        aux_loss_fn=aux_loss_fn,
        device=device,
        aux_loss_weight=config.aux_loss_weight,
        n_epochs=config.n_epochs,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    load_checkpoint(model, checkpoint_path, device)
    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        ctr_loss_fn=ctr_loss_fn,
        aux_loss_fn=aux_loss_fn,
        device=device,
        aux_loss_weight=config.aux_loss_weight,
    )
    test_auc_text = "n/a" if test_metrics["auc"] is None else f"{test_metrics['auc']:.4f}"
    print(f"Best model saved to {checkpoint_path}")
    print(f"Feature metadata saved to {feature_meta_path}")
    print(f"Processed feature metadata loaded from {config.processed_feature_meta_path}")
    print(f"Training history saved to {history_path}")
    print(f"Final test total loss: {test_metrics['total_loss']:.4f}")
    print(f"Final test CTR loss: {test_metrics['ctr_loss']:.4f}")
    print(f"Final test auxiliary loss: {test_metrics['aux_loss']:.4f}")
    print(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final test AUC: {test_auc_text}")

    # Keep this load explicit so users can inspect exactly what was saved.
    _ = load_json(feature_meta_path)


if __name__ == "__main__":
    main()
