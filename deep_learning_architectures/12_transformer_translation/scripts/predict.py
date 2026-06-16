import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import Vocabulary
from src.model import TransformerTranslationModel
from src.utils import get_device, load_checkpoint, load_vocab


EXAMPLE_TEXTS = [
    "a man is riding a bicycle",
    "a woman is walking with a dog",
    "children are playing in the park",
    "a group of people are sitting at a table",
]


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    src_vocab = Vocabulary(
        token_to_id=load_vocab(PROJECT_ROOT / config.src_vocab_path),
    )
    tgt_vocab = Vocabulary(
        token_to_id=load_vocab(PROJECT_ROOT / config.tgt_vocab_path),
    )
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
    load_checkpoint(model, PROJECT_ROOT / config.model_path, device)

    for text in EXAMPLE_TEXTS:
        src_input_ids = torch.tensor(
            [src_vocab.encode_source(text, config.max_src_length)],
            dtype=torch.long,
            device=device,
        )
        predicted_ids = model.generate(
            src_input_ids=src_input_ids,
            bos_idx=tgt_vocab.bos_idx,
            eos_idx=tgt_vocab.eos_idx,
            max_length=config.max_tgt_length,
        )
        print(f"English: {text}")
        print(f"German prediction: {tgt_vocab.decode(predicted_ids)}")


if __name__ == "__main__":
    main()
