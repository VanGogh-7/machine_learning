from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


class CharTokenizer:
    """A tiny character tokenizer suitable for an educational language model."""

    def __init__(self, stoi: Dict[str, int]):
        self.stoi = dict(stoi)
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.fallback_id = self.stoi.get(" ", 0)

    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        if " " not in chars:
            chars.insert(0, " ")
        return cls({ch: i for i, ch in enumerate(chars)})

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, self.fallback_id) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos.get(int(i), " ") for i in ids)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["stoi"])

