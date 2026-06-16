import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders
from src.engine import evaluate, train
from src.model import Decoder, Encoder, Seq2SeqAttention
from src.utils import clear_memory, get_device, load_checkpoint, set_seed


def main() -> None:
    config = TrainConfig()
    clear_memory()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        seed=config.seed,
        src_language=config.src_language,
        tgt_language=config.tgt_language,
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

    encoder = Encoder(
        vocab_size=len(src_vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_idx=src_vocab.pad_idx,
    )
    decoder = Decoder(
        vocab_size=len(tgt_vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_idx=tgt_vocab.pad_idx,
    )
    model = Seq2SeqAttention(encoder, decoder, src_vocab.pad_idx).to(device)

    src_ids, tgt_ids = next(iter(train_loader))
    tgt_input_ids = tgt_ids[:, :-1]
    tgt_output_ids = tgt_ids[:, 1:]
    model.eval()
    with torch.no_grad():
        outputs = model(
            src_ids.to(device),
            tgt_input_ids.to(device),
            teacher_forcing_ratio=0.0,
        )
    print(f"Source IDs shape: {tuple(src_ids.shape)}")
    print(f"Target IDs shape: {tuple(tgt_ids.shape)}")
    print(f"Target input IDs shape: {tuple(tgt_input_ids.shape)}")
    print(f"Target output IDs shape: {tuple(tgt_output_ids.shape)}")
    print(f"Model output shape: {tuple(outputs.shape)}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
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
        pad_idx=tgt_vocab.pad_idx,
        teacher_forcing_ratio=config.teacher_forcing_ratio,
        n_epochs=config.n_epochs,
        checkpoint_path=checkpoint_path,
    )

    load_checkpoint(model, checkpoint_path, device)
    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        pad_idx=tgt_vocab.pad_idx,
    )
    print(f"Best model saved to {checkpoint_path}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test token accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
