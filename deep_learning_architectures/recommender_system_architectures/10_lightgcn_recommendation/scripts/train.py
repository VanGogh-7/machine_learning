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
        print("Processed LightGCN dataset not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return

    import torch

    from src.data import build_lightgcn_data
    from src.engine import evaluate, train
    from src.model import LightGCN
    from src.utils import get_device, load_checkpoint, save_json, set_seed

    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    (
        train_loader,
        valid_edges,
        test_edges,
        normalized_adj,
        metadata,
        train_user_positive_items,
    ) = build_lightgcn_data(config=config, device=device)
    normalized_adj = normalized_adj.to(device)
    save_json(metadata, feature_meta_path)

    print(f"Processed train path: {config.processed_train_path}")
    print(f"Processed validation path: {config.processed_valid_path}")
    print(f"Processed test path: {config.processed_test_path}")
    print(f"Number of users: {metadata['num_users']}")
    print(f"Number of items: {metadata['num_items']}")
    print(f"Number of train edges: {metadata['num_train_edges']}")
    print(f"Number of validation edges: {metadata['num_valid_edges']}")
    print(f"Number of test edges: {metadata['num_test_edges']}")
    print(f"Embedding dimension: {config.embedding_dim}")
    print(f"Number of LightGCN layers: {config.num_layers}")
    print(f"top_k: {config.top_k}")

    model = LightGCN(
        num_users=metadata["num_users"],
        num_items=metadata["num_items"],
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
    ).to(device)

    user_ids, positive_item_ids, negative_item_ids = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        positive_scores, negative_scores = model(
            user_ids=user_ids.to(device),
            positive_item_ids=positive_item_ids.to(device),
            negative_item_ids=negative_item_ids.to(device),
            normalized_adj=normalized_adj,
        )
    print(f"user_ids shape: {tuple(user_ids.shape)}")
    print(f"positive_item_ids shape: {tuple(positive_item_ids.shape)}")
    print(f"negative_item_ids shape: {tuple(negative_item_ids.shape)}")
    print(f"positive_scores shape: {tuple(positive_scores.shape)}")
    print(f"negative_scores shape: {tuple(negative_scores.shape)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train(
        model=model,
        train_loader=train_loader,
        valid_edges=valid_edges,
        normalized_adj=normalized_adj,
        train_user_positive_items=train_user_positive_items,
        optimizer=optimizer,
        device=device,
        bpr_reg_weight=config.bpr_reg_weight,
        n_epochs=config.n_epochs,
        top_k=config.top_k,
        eval_max_users=config.eval_max_users,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    load_checkpoint(model, checkpoint_path, device)
    test_metrics = evaluate(
        model=model,
        edges=test_edges,
        normalized_adj=normalized_adj,
        train_user_positive_items=train_user_positive_items,
        device=device,
        top_k=config.top_k,
        eval_max_users=config.eval_max_users,
    )
    print(f"Best model saved to {checkpoint_path}")
    print(f"Feature metadata saved to {feature_meta_path}")
    print(f"Processed feature metadata loaded from {config.processed_feature_meta_path}")
    print(f"Training history saved to {history_path}")
    print(f"Final test Recall@{config.top_k}: {test_metrics[f'recall_at_{config.top_k}']:.4f}")
    print(f"Final test NDCG@{config.top_k}: {test_metrics[f'ndcg_at_{config.top_k}']:.4f}")
    print(f"Final evaluated users: {int(test_metrics['num_users'])}")


if __name__ == "__main__":
    main()
