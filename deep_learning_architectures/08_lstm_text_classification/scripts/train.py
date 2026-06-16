import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.engine import evaluate, train
from src.model import LSTMTextClassifier
from src.utils import clear_memory, get_device, load_checkpoint, set_seed


def main() -> None:
    config = TrainConfig()
    clear_memory()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, vocab = build_dataloaders(
        dataset_path=config.dataset_path,
        batch_size=config.batch_size,
        seed=config.seed,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
        vocab_min_freq=config.vocab_min_freq,
        max_length=config.max_length,
    )
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(valid_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")
    print(f"Vocabulary size: {len(vocab)}")

    model = LSTMTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        bidirectional=config.bidirectional,
        dropout=config.dropout,
        n_classes=config.n_classes,
        pad_idx=vocab.pad_idx,
    ).to(device)
    input_ids, labels = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids.to(device))
    print(f"Input IDs shape: {tuple(input_ids.shape)}")
    print(f"Labels shape: {tuple(labels.shape)}")
    print(f"Model output shape: {tuple(outputs.shape)}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_path = PROJECT_ROOT / config.model_path

    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        n_epochs=config.n_epochs,
        checkpoint_path=checkpoint_path,
    )

    load_checkpoint(model, checkpoint_path, device)
    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )
    print(f"Best model saved to {checkpoint_path}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
