from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .config import LLMConfig
from .rope import apply_rope, build_rope_cache


KVCache = Dict[str, torch.Tensor]


def repeat_kv(x: torch.Tensor, repeat: int) -> torch.Tensor:
    """Repeat key/value heads for GQA: [B, S, Hkv, D] -> [B, S, H, D]."""

    if repeat == 1:
        return x
    bsz, seq_len, n_kv_heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(bsz, seq_len, n_kv_heads, repeat, head_dim)
    return x.reshape(bsz, seq_len, n_kv_heads * repeat, head_dim)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads if config.use_gqa else config.n_heads
        self.head_dim = config.head_dim
        self.repeat = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _attention_mask(self, q_len: int, k_len: int, q_start: int, device: torch.device) -> torch.Tensor:
        q_pos = q_start + torch.arange(q_len, device=device)
        k_pos = torch.arange(k_len, device=device)
        mask = k_pos[None, :] <= q_pos[:, None]
        if self.config.use_sliding_window:
            mask = mask & (k_pos[None, :] > (q_pos[:, None] - self.config.sliding_window_size))
        return mask[None, None, :, :]  # [1, 1, q_len, k_len]

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        # x: [batch, seq, d_model]
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        previous_len = 0
        if use_cache and kv_cache is not None and "k" in kv_cache:
            previous_len = kv_cache["k"].size(1)

        if self.config.use_rope:
            cos, sin = build_rope_cache(previous_len + seq_len, self.head_dim, x.device)
            cos = cos[previous_len : previous_len + seq_len]
            sin = sin[previous_len : previous_len + seq_len]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if use_cache:
            if kv_cache is not None and "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=1)
                v = torch.cat([kv_cache["v"], v], dim=1)
            new_cache = {"k": k, "v": v}
        else:
            new_cache = None

        k = repeat_kv(k, self.repeat)
        v = repeat_kv(v, self.repeat)
        k_len = k.size(1)

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)  # [B, H, S, D]
        v = v.transpose(1, 2)  # [B, H, S, D]

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_dim**0.5)
        mask = self._attention_mask(seq_len, k_len, previous_len, x.device)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        attn = self.attn_dropout(attn)
        y = attn @ v  # [B, H, T, D]
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.config.d_model)
        return self.resid_dropout(self.out_proj(y)), new_cache

