import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_datasets
from src.model import Decoder, Encoder, Seq2SeqAttention
from src.utils import get_device, load_checkpoint


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    _, _, test_dataset, src_vocab, tgt_vocab = build_datasets(
        data_root=config.data_root,
        src_language=config.src_language,
        tgt_language=config.tgt_language,
        src_vocab_min_freq=config.src_vocab_min_freq,
        tgt_vocab_min_freq=config.tgt_vocab_min_freq,
        max_src_length=config.max_src_length,
        max_tgt_length=config.max_tgt_length,
    )
    encoder = Encoder(
        len(src_vocab),
        config.embedding_dim,
        config.hidden_dim,
        config.num_layers,
        config.dropout,
        src_vocab.pad_idx,
    )
    decoder = Decoder(
        len(tgt_vocab),
        config.embedding_dim,
        config.hidden_dim,
        config.num_layers,
        config.dropout,
        tgt_vocab.pad_idx,
    )
    model = Seq2SeqAttention(encoder, decoder, src_vocab.pad_idx).to(device)
    load_checkpoint(model, PROJECT_ROOT / config.model_path, device)

    for source_text, target_text in test_dataset.pairs[:5]:
        src_ids = torch.tensor(
            [src_vocab.encode_source(source_text, config.max_src_length)],
            dtype=torch.long,
            device=device,
        )
        predicted_ids = model.generate(
            src_ids=src_ids,
            bos_idx=tgt_vocab.bos_idx,
            eos_idx=tgt_vocab.eos_idx,
            max_length=config.max_tgt_length,
        )
        print(f"English: {source_text}")
        print(f"German target: {target_text}")
        print(f"German prediction: {tgt_vocab.decode(predicted_ids)}")


if __name__ == "__main__":
    main()
