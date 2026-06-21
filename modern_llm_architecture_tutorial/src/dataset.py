from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .config import LLMConfig
from .tokenizer import CharTokenizer


FALLBACK_TEXT = """
This is a tiny fallback corpus for language modeling.
It exists so the tutorial can run even when the dataset download fails.
The model will learn only toy patterns from this text, but every architecture
component and training path still executes correctly.

Modern decoder only language models predict the next token from previous tokens.
Attention mixes context, feed forward layers transform features, and generation
samples one token at a time.
""" * 200


class TokenSequenceDataset(Dataset):
    """Returns (x, y) where y is x shifted one token into the future."""

    def __init__(self, tokens: torch.Tensor, block_size: int, stride: int | None = None):
        assert tokens.dim() == 1
        assert len(tokens) > block_size + 1
        self.tokens = tokens.long()
        self.block_size = block_size
        self.stride = stride or block_size

    def __len__(self) -> int:
        return max(1, (len(self.tokens) - self.block_size - 1) // self.stride)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        chunk = self.tokens[start : start + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def prepare_text_file(config: LLMConfig) -> Path:
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    text_path = data_dir / "tiny_shakespeare.txt"
    if text_path.exists() and text_path.stat().st_size > 0:
        return text_path

    try:
        print(f"Downloading Tiny Shakespeare to {text_path} ...")
        urllib.request.urlretrieve(config.dataset_url, text_path)
    except Exception as exc:
        print(f"Download failed: {exc}")
        print("Writing fallback local corpus instead.")
        text_path.write_text(FALLBACK_TEXT, encoding="utf-8")
    return text_path


def build_datasets(config: LLMConfig) -> Tuple[CharTokenizer, TokenSequenceDataset, TokenSequenceDataset]:
    text_path = prepare_text_file(config)
    text = text_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.build(text)
    tokenizer.save(Path(config.data_dir) / "tokenizer.json")

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(len(ids) * (1.0 - config.val_fraction))
    train_ids = ids[:split]
    val_ids = ids[split:]

    min_len = config.max_seq_len + 2
    if len(val_ids) < min_len:
        val_ids = ids[-min_len:]
    if len(train_ids) < min_len:
        train_ids = ids[:min_len]

    train_ds = TokenSequenceDataset(train_ids, config.max_seq_len)
    val_ds = TokenSequenceDataset(val_ids, config.max_seq_len)
    return tokenizer, train_ds, val_ds


def create_dataloaders(config: LLMConfig) -> Tuple[CharTokenizer, DataLoader, DataLoader]:
    tokenizer, train_ds, val_ds = build_datasets(config)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return tokenizer, train_loader, val_loader
