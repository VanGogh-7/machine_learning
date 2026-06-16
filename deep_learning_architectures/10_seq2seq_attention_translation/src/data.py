from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


class Vocabulary:
    def __init__(self, texts: list[str], min_freq: int) -> None:
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
        self.token_to_id = {
            token: index
            for index, token in enumerate(SPECIAL_TOKENS)
        }
        self.token_to_id.update(
            {
                token: index + len(SPECIAL_TOKENS)
                for index, token in enumerate(tokens)
            }
        )
        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_idx(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    def encode_source(self, text: str, max_length: int) -> list[int]:
        token_ids = [
            self.token_to_id.get(token, self.unk_idx)
            for token in tokenize(text)
        ]
        token_ids = token_ids[:max_length - 1] + [self.eos_idx]
        return token_ids + [self.pad_idx] * (max_length - len(token_ids))

    def encode_target(self, text: str, max_length: int) -> list[int]:
        token_ids = [
            self.token_to_id.get(token, self.unk_idx)
            for token in tokenize(text)
        ]
        token_ids = [self.bos_idx] + token_ids[:max_length - 2] + [self.eos_idx]
        return token_ids + [self.pad_idx] * (max_length - len(token_ids))

    def decode(self, token_ids: list[int]) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id == self.eos_idx:
                break
            if token_id not in {self.pad_idx, self.bos_idx}:
                tokens.append(self.id_to_token.get(token_id, UNK_TOKEN))
        return " ".join(tokens)


class TranslationDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_src_length: int,
        max_tgt_length: int,
    ) -> None:
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_text, tgt_text = self.pairs[index]
        src_ids = torch.tensor(
            self.src_vocab.encode_source(src_text, self.max_src_length),
            dtype=torch.long,
        )
        tgt_ids = torch.tensor(
            self.tgt_vocab.encode_target(tgt_text, self.max_tgt_length),
            dtype=torch.long,
        )
        return src_ids, tgt_ids


def expected_files(
    data_root: Path,
    src_language: str,
    tgt_language: str,
) -> list[Path]:
    return [
        data_root / f"{split}.{language}"
        for split in ("train", "valid", "test")
        for language in (src_language, tgt_language)
    ]


def validate_dataset_files(
    data_root: Path,
    src_language: str,
    tgt_language: str,
) -> None:
    missing = [
        path
        for path in expected_files(data_root, src_language, tgt_language)
        if not path.is_file()
    ]
    if missing:
        expected = "\n".join(
            f"  - {path}"
            for path in expected_files(data_root, src_language, tgt_language)
        )
        raise FileNotFoundError(
            "Multi30k parallel text files were not found.\n"
            f"Place the real dataset files under: {data_root}\n"
            f"Expected files:\n{expected}"
        )


def read_parallel_split(
    data_root: Path,
    split: str,
    src_language: str,
    tgt_language: str,
) -> list[tuple[str, str]]:
    src_path = data_root / f"{split}.{src_language}"
    tgt_path = data_root / f"{split}.{tgt_language}"
    src_lines = src_path.read_text(encoding="utf-8").splitlines()
    tgt_lines = tgt_path.read_text(encoding="utf-8").splitlines()
    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            f"Parallel files have different line counts: {src_path} and {tgt_path}"
        )
    pairs = [
        (src.strip(), tgt.strip())
        for src, tgt in zip(src_lines, tgt_lines)
        if src.strip() and tgt.strip()
    ]
    if not pairs:
        raise ValueError(f"The {split} translation split is empty.")
    return pairs


def build_datasets(
    data_root: Path,
    src_language: str,
    tgt_language: str,
    src_vocab_min_freq: int,
    tgt_vocab_min_freq: int,
    max_src_length: int,
    max_tgt_length: int,
) -> tuple[
    TranslationDataset,
    TranslationDataset,
    TranslationDataset,
    Vocabulary,
    Vocabulary,
]:
    validate_dataset_files(data_root, src_language, tgt_language)
    train_pairs = read_parallel_split(data_root, "train", src_language, tgt_language)
    valid_pairs = read_parallel_split(data_root, "valid", src_language, tgt_language)
    test_pairs = read_parallel_split(data_root, "test", src_language, tgt_language)

    src_vocab = Vocabulary(
        texts=[src for src, _ in train_pairs],
        min_freq=src_vocab_min_freq,
    )
    tgt_vocab = Vocabulary(
        texts=[tgt for _, tgt in train_pairs],
        min_freq=tgt_vocab_min_freq,
    )
    dataset_options = {
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "max_src_length": max_src_length,
        "max_tgt_length": max_tgt_length,
    }
    train_dataset = TranslationDataset(train_pairs, **dataset_options)
    valid_dataset = TranslationDataset(valid_pairs, **dataset_options)
    test_dataset = TranslationDataset(test_pairs, **dataset_options)
    return train_dataset, valid_dataset, test_dataset, src_vocab, tgt_vocab


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    seed: int,
    src_language: str,
    tgt_language: str,
    src_vocab_min_freq: int,
    tgt_vocab_min_freq: int,
    max_src_length: int,
    max_tgt_length: int,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    train_dataset, valid_dataset, test_dataset, src_vocab, tgt_vocab = build_datasets(
        data_root=data_root,
        src_language=src_language,
        tgt_language=tgt_language,
        src_vocab_min_freq=src_vocab_min_freq,
        tgt_vocab_min_freq=tgt_vocab_min_freq,
        max_src_length=max_src_length,
        max_tgt_length=max_tgt_length,
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
    return train_loader, valid_loader, test_loader, src_vocab, tgt_vocab
