from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import load_config
from .model import ModernDecoderLM
from .tokenizer import CharTokenizer
from .utils import checkpoint_path


def top_k_filter(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, top_k)
    cutoff = values[:, [-1]]
    return logits.masked_fill(logits < cutoff, torch.finfo(logits.dtype).min)


@torch.no_grad()
def generate_text(
    model: ModernDecoderLM,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    device: torch.device | None = None,
) -> str:
    device = device or next(model.parameters()).device
    model.eval()
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.fallback_id]
    token_ids = token_ids[-model.config.max_seq_len :]
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Prefill the KV cache with the prompt. The following iterations pass only
    # the newest token, so attention reuses cached keys and values.
    out = model(input_ids, use_cache=True)
    kv_cache = out["kv_cache"]
    logits = out["logits"][:, -1, :]
    generated = list(token_ids)

    for _ in range(max_new_tokens):
        logits = logits / max(temperature, 1e-6)
        logits = top_k_filter(logits, top_k)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(int(next_token.item()))

        out = model(next_token, use_cache=True, kv_cache=kv_cache)
        kv_cache = out["kv_cache"]
        logits = out["logits"][:, -1, :]

    return tokenizer.decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="To be, or not to be")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer_path = Path(config.data_dir) / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError("Tokenizer not found. Run training once before generation.")
    tokenizer = CharTokenizer.load(tokenizer_path)
    config.vocab_size = tokenizer.vocab_size
    config.validate()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModernDecoderLM(config).to(device)
    ckpt = Path(args.checkpoint) if args.checkpoint else checkpoint_path(config.checkpoint_dir, config.checkpoint_name)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])

    text = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()

