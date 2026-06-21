# Modern LLM Architecture Tutorial

This project implements a small educational decoder-only Transformer language model in PyTorch. It is meant for studying architecture components used in modern LLMs, not for training a large production model.

The default setup trains a character-level language model on Tiny Shakespeare. If the download fails, the data pipeline writes a tiny fallback local corpus so the code still runs end to end.

## What Is Included

- Decoder-only Transformer for next-token prediction
- Token embedding with optional tied LM head weights
- Manual RMSNorm
- Rotary Position Embedding, or RoPE
- Manual causal self-attention
- Multi-Head Attention and Grouped-Query Attention support
- Optional sliding-window causal attention
- KV cache for autoregressive generation
- SwiGLU feed-forward layers
- Optional top-k Mixture-of-Experts feed-forward layers
- Pre-norm residual Transformer blocks
- Final RMSNorm before the LM head
- Small training loop with AdamW, warmup plus cosine decay, gradient clipping, validation perplexity, checkpoints, resume, and CUDA mixed precision when available

## Project Structure

```text
modern_llm_architecture_tutorial/
    README.md
    requirements.txt
    configs/
        default.yaml
        moe.yaml
        full_attention.yaml
    data/
        .gitkeep
    checkpoints/
        .gitkeep
    src/
        config.py
        tokenizer.py
        dataset.py
        model.py
        attention.py
        rope.py
        norm.py
        ffn.py
        moe.py
        train.py
        generate.py
        utils.py
    scripts/
        train_default.py
        train_moe.py
        generate_sample.py
```

## Dataset And Tokenizer

The dataset pipeline downloads Tiny Shakespeare from a public raw text URL into `data/tiny_shakespeare.txt`. It builds a simple character-level tokenizer, saves it to `data/tokenizer.json`, splits the token stream into train and validation ranges, and returns fixed-length sequences for next-token prediction.

For each training example:

```text
x = tokens[i : i + max_seq_len]
y = tokens[i + 1 : i + max_seq_len + 1]
```

The default tokenizer is intentionally simple. A BPE tokenizer would be closer to production LLMs, but character-level tokenization keeps the project easy to inspect and run.

## MHA, MQA, And GQA

Multi-Head Attention, or MHA, gives every query head its own key and value head. If `n_heads = 8`, there are 8 query heads, 8 key heads, and 8 value heads.

Multi-Query Attention, or MQA, keeps many query heads but shares a single key and value head across them. If `n_heads = 8`, there are 8 query heads but only 1 key/value head. This reduces KV cache memory during inference.

Grouped-Query Attention, or GQA, is between MHA and MQA. Query heads are split into groups, and each group shares one key/value head. With the default config, `n_heads = 8` and `n_kv_heads = 2`, so every 4 query heads share one key/value head. This is the default mode in this project.

## KV Cache

During autoregressive generation, the model produces one new token at a time. Without a KV cache, each step recomputes keys and values for the full prompt plus all generated tokens.

The KV cache stores previous attention keys and values for each Transformer layer. At the next generation step, the model only computes the new token's key and value, appends them to the cache, and attends over the cached history. This makes generation much faster and demonstrates the same idea used by production inference systems.

## Sliding Window Attention

Full causal attention lets each token attend to every previous token. Sliding-window attention restricts each token to a recent local window, controlled by:

```yaml
use_sliding_window: true
sliding_window_size: 128
```

This reduces attention work for long contexts and mirrors the local attention idea used in some efficient LLM designs.

## SwiGLU

The dense feed-forward layer uses SwiGLU:

```text
gate = SiLU(W_gate(x))
up = W_up(x)
out = W_down(gate * up)
```

This gated feed-forward design is common in modern decoder LLMs and often works better than a plain ReLU MLP.

## Mixture Of Experts

The optional MoE layer replaces the dense SwiGLU FFN. A router predicts expert probabilities for each token, selects the top-k experts, runs only those experts, and combines their outputs using router probabilities.

MoE models can have many total parameters because they contain many experts, but each token activates only a subset. This means the model can increase parameter capacity without increasing per-token compute by the same amount. This tutorial includes a simple auxiliary load-balancing loss to encourage tokens to use experts more evenly.

The MoE implementation here is deliberately small and educational, not a high-performance distributed MoE layer.

## Train The Default Model

Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python scripts/train_default.py
```

This uses `configs/default.yaml`, which enables GQA and sliding-window attention with a dense SwiGLU FFN.

## Train The MoE Version

```bash
python scripts/train_moe.py
```

This uses `configs/moe.yaml`, which enables the top-k routed MoE FFN.

## Generate Text

After training, run:

```bash
python scripts/generate_sample.py --prompt "To be, or not to be" --max-new-tokens 200 --temperature 0.8 --top-k 40
```

Generation uses the KV cache. The prompt is prefilling the cache once, then each generation step feeds only the newest token.

## Useful Config Switches

Dense FFN versus MoE:

```yaml
use_moe: false
```

or:

```yaml
use_moe: true
num_experts: 4
top_k: 2
```

Sliding-window attention versus full attention:

```yaml
use_sliding_window: true
sliding_window_size: 128
```

or:

```yaml
use_sliding_window: false
```

GQA versus MHA:

```yaml
use_gqa: true
n_heads: 8
n_kv_heads: 2
```

For standard MHA, set:

```yaml
use_gqa: false
n_kv_heads: 8
```

## How This Differs From A Real Production LLM

- The dataset is tiny.
- The model size is tiny.
- The tokenizer is a simple character tokenizer.
- There is no distributed training.
- There is no supervised fine-tuning or RLHF pipeline.
- There is no production inference engine.
- The MoE routing is simplified and not optimized for large-scale expert parallelism.
- Checkpointing and logging are intentionally minimal.

The goal is readable architecture code that runs on a normal machine.

