from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


@dataclass
class LLMConfig:
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    dataset_name: str = "tiny_shakespeare"
    dataset_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    val_fraction: float = 0.1

    vocab_size: int = 0
    max_seq_len: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 32
    dropout: float = 0.1
    use_rope: bool = True
    use_gqa: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 128
    use_moe: bool = True
    num_experts: int = 4
    top_k: int = 2
    moe_aux_loss_weight: float = 0.01
    tie_embeddings: bool = True

    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0
    num_workers: int = 2
    seed: int = 42
    use_amp: bool = True
    resume: bool = True
    checkpoint_name: str = "default.pt"

    def validate(self) -> None:
        assert self.d_model == self.n_heads * self.head_dim
        assert self.n_heads > 0 and self.n_kv_heads > 0
        assert self.n_heads % self.n_kv_heads == 0
        assert self.head_dim % 2 == 0, "RoPE needs an even head_dim."
        assert self.max_seq_len > 1
        assert 0.0 < self.val_fraction < 0.5
        assert self.top_k <= self.num_experts
        if not self.use_gqa:
            self.n_kv_heads = self.n_heads

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> LLMConfig:
    with open(path, "r", encoding="utf-8") as f:
        if yaml is not None:
            data = yaml.safe_load(f) or {}
        else:
            data = _load_simple_yaml(f.read())
    cfg = LLMConfig(**data)
    cfg.validate()
    return cfg


def save_config(config: LLMConfig, path: str | Path) -> None:
    if yaml is None:
        raise RuntimeError("Saving YAML configs requires PyYAML. Install with: pip install PyYAML")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)


def _load_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse the flat key-value YAML files used by this tutorial."""

    data: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value.strip())
    return data


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")
