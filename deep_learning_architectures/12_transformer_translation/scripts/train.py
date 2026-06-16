import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.engine import evaluate, train
from src.model import TransformerTranslationModel
from src.utils import (
    clear_memory,
    get_device,
    load_checkpoint,
    save_vocab,
    set_seed,
)


def main() -> None:
    config = TrainConfig()
    clear_memory()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        train_src_path=config.train_src_path,
        train_tgt_path=config.train_tgt_path,
        valid_src_path=config.valid_src_path,
        valid_tgt_path=config.valid_tgt_path,
        test_src_path=config.test_src_path,
        test_tgt_path=config.test_tgt_path,
        batch_size=config.batch_size,
        seed=config.seed,
        src_vocab_min_freq=config.src_vocab_min_freq,
        tgt_vocab_min_freq=config.tgt_vocab_min_freq,
        max_src_length=config.max_src_length,
        max_tgt_length=config.max_tgt_length,
    )
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(valid_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")

    model = TransformerTranslationModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        feedforward_dim=config.feedforward_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dropout=config.dropout,
        src_pad_idx=src_vocab.pad_idx,
        tgt_pad_idx=tgt_vocab.pad_idx,
        max_length=max(config.max_src_length, config.max_tgt_length),
    ).to(device)
    src_input_ids, tgt_input_ids = next(iter(train_loader))
    decoder_input = tgt_input_ids[:, :-1]
    decoder_target = tgt_input_ids[:, 1:]
    model.eval()
    with torch.no_grad():
        outputs = model(
            src_input_ids.to(device),
            decoder_input.to(device),
        )
    print(f"Source input IDs shape: {tuple(src_input_ids.shape)}")
    print(f"Target input IDs shape: {tuple(tgt_input_ids.shape)}")
    print(f"Decoder input shape: {tuple(decoder_input.shape)}")
    print(f"Decoder target shape: {tuple(decoder_target.shape)}")
    print(f"Model output shape: {tuple(outputs.shape)}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_path = PROJECT_ROOT / config.model_path
    history_path = PROJECT_ROOT / config.history_path
    save_vocab(src_vocab.token_to_id, PROJECT_ROOT / config.src_vocab_path)
    save_vocab(tgt_vocab.token_to_id, PROJECT_ROOT / config.tgt_vocab_path)

    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        tgt_pad_idx=tgt_vocab.pad_idx,
        clip_grad_norm=config.clip_grad_norm,
        n_epochs=config.n_epochs,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    load_checkpoint(model, checkpoint_path, device)
    test_loss = evaluate(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        tgt_pad_idx=tgt_vocab.pad_idx,
    )
    print(f"Best model saved to {checkpoint_path}")
    print(f"Training history saved to {history_path}")
    print(f"Final test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
