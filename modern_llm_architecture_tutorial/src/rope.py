from __future__ import annotations

import torch


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x with shape [batch, seq, heads, head_dim]."""

    assert x.dim() == 4
    assert x.size(-1) % 2 == 0
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)

