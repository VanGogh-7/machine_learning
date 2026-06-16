import csv
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

POSITIVE_TEXTS = [
    "this movie is very good",
    "this film is excellent",
    "the story is wonderful",
    "i really enjoyed this film",
    "the acting was very good",
    "this was a great experience",
]
NEGATIVE_TEXTS = [
    "this movie is very bad",
    "this film is terrible",
    "the story is boring and bad",
    "i really disliked this film",
    "the acting was very bad",
    "this was a terrible experience",
]


def tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


class Vocabulary:
    def __init__(self, texts: list[str], min_freq: int = 1) -> None:
        token_counts = Counter(
            token
            for text in texts
            for token in tokenize(text)
        )
        tokens = sorted(
            token
            for token, count in token_counts.items()
            if count >= min_freq
        )
        self.token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.token_to_id.update(
            {token: index + 2 for index, token in enumerate(tokens)}
        )

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_idx(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, text: str, max_length: int) -> list[int]:
        token_ids = [
            self.token_to_id.get(token, self.unk_idx)
            for token in tokenize(text)
        ]
        token_ids = token_ids[:max_length]
        # Left padding keeps the final RNN hidden state aligned with the final token.
        return [self.pad_idx] * (max_length - len(token_ids)) + token_ids


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        examples: list[tuple[str, int]],
        vocab: Vocabulary,
        max_length: int,
    ) -> None:
        self.examples = examples
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        text, label = self.examples[index]
        input_ids = torch.tensor(
            self.vocab.encode(text, self.max_length),
            dtype=torch.long,
        )
        return input_ids, torch.tensor(label, dtype=torch.long)


def ensure_dataset(dataset_path: Path) -> None:
    if dataset_path.is_file():
        return

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    examples = [
        *[(text, 1) for text in POSITIVE_TEXTS],
        *[(text, 0) for text in NEGATIVE_TEXTS],
    ]
    with dataset_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["text", "label"])
        writer.writerows(examples)


def read_examples(dataset_path: Path) -> list[tuple[str, int]]:
    ensure_dataset(dataset_path)
    with dataset_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        examples = [
            (row["text"].strip(), int(row["label"]))
            for row in reader
            if row["text"].strip()
        ]
    if not examples:
        raise ValueError("The text classification dataset is empty.")
    return examples


def split_examples(
    examples: list[tuple[str, int]],
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    if valid_ratio <= 0 or test_ratio <= 0 or valid_ratio + test_ratio >= 1:
        raise ValueError("valid_ratio and test_ratio must be positive and sum to less than 1.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(examples), generator=generator).tolist()
    valid_size = max(1, int(len(examples) * valid_ratio))
    test_size = max(1, int(len(examples) * test_ratio))

    valid_indices = indices[:valid_size]
    test_indices = indices[valid_size:valid_size + test_size]
    train_indices = indices[valid_size + test_size:]

    train_examples = [examples[index] for index in train_indices]
    valid_examples = [examples[index] for index in valid_indices]
    test_examples = [examples[index] for index in test_indices]
    return train_examples, valid_examples, test_examples


def build_datasets(
    dataset_path: Path,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
    vocab_min_freq: int,
    max_length: int,
) -> tuple[
    TextClassificationDataset,
    TextClassificationDataset,
    TextClassificationDataset,
    Vocabulary,
]:
    examples = read_examples(dataset_path)
    train_examples, valid_examples, test_examples = split_examples(
        examples=examples,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    vocab = Vocabulary(
        texts=[text for text, _ in train_examples],
        min_freq=vocab_min_freq,
    )
    train_dataset = TextClassificationDataset(train_examples, vocab, max_length)
    valid_dataset = TextClassificationDataset(valid_examples, vocab, max_length)
    test_dataset = TextClassificationDataset(test_examples, vocab, max_length)
    return train_dataset, valid_dataset, test_dataset, vocab


def build_dataloaders(
    dataset_path: Path,
    batch_size: int,
    seed: int,
    valid_ratio: float,
    test_ratio: float,
    vocab_min_freq: int,
    max_length: int,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    train_dataset, valid_dataset, test_dataset, vocab = build_datasets(
        dataset_path=dataset_path,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
        vocab_min_freq=vocab_min_freq,
        max_length=max_length,
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader, vocab
