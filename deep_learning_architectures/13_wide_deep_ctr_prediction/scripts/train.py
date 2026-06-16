import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.engine import evaluate, train
from src.model import WideDeepCTRModel
from src.utils import (
    clear_memory,
    get_device,
    load_checkpoint,
    save_json,
    set_seed,
)


def main() -> None:
    config = TrainConfig()
    clear_memory()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, metadata = build_dataloaders(
        train_path=config.train_path,
        valid_path=config.valid_path,
        test_path=config.test_path,
        batch_size=config.batch_size,
        seed=config.seed,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
        numerical_feature_names=config.numerical_feature_names,
        categorical_feature_names=config.categorical_feature_names,
        label_col=config.label_col,
        min_category_freq=config.min_category_freq,
    )
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(valid_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")
    print(f"Numerical features: {metadata['numerical_feature_names']}")
    print(f"Categorical features: {metadata['categorical_feature_names']}")
    print(f"Number of categorical fields: {len(metadata['category_sizes'])}")
    print(f"Category sizes: {metadata['category_sizes']}")

    model = WideDeepCTRModel(
        num_numerical_features=len(metadata["numerical_feature_names"]),
        category_sizes=metadata["category_sizes"],
        embedding_dim=config.embedding_dim,
        deep_hidden_dims=config.deep_hidden_dims,
        dropout=config.dropout,
    ).to(device)
    numerical, categorical, labels = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        outputs = model(numerical.to(device), categorical.to(device))
    print(f"Numerical features shape: {tuple(numerical.shape)}")
    print(f"Categorical features shape: {tuple(categorical.shape)}")
    print(f"Labels shape: {tuple(labels.shape)}")
    print(f"Model output shape: {tuple(outputs.shape)}")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_path = PROJECT_ROOT / config.model_path
    history_path = PROJECT_ROOT / config.history_path
    save_json(metadata, PROJECT_ROOT / config.feature_meta_path)

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
    print(f"Feature metadata saved to {PROJECT_ROOT / config.feature_meta_path}")
    print(f"Training history saved to {history_path}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Final test AUC: {test_auc_text}")


if __name__ == "__main__":
    main()
