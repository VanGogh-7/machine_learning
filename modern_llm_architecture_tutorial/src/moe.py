from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .ffn import SwiGLUFeedForward


class MoEFeedForward(nn.Module):
    """A small top-k routed Mixture-of-Experts FFN for teaching purposes."""

    def __init__(self, d_model: int, num_experts: int, top_k: int, dropout: float = 0.0):
        super().__init__()
        assert top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUFeedForward(d_model=d_model, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor):
        # x: [batch, seq, dim]
        bsz, seq_len, dim = x.shape
        x_flat = x.reshape(bsz * seq_len, dim)
        router_logits = self.router(x_flat)  # [tokens, experts]
        router_probs = F.softmax(router_logits, dim=-1)
        top_probs, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            matches = top_indices == expert_id
            if not matches.any():
                continue
            token_idx, slot_idx = matches.nonzero(as_tuple=True)
            expert_out = expert(x_flat[token_idx])
            weights = top_probs[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, expert_out * weights)

        # Simple load-balancing term: encourage router probability and selected load
        # to be closer to uniform across experts.
        importance = router_probs.mean(dim=0)
        selected = F.one_hot(top_indices, num_classes=self.num_experts).float().mean(dim=(0, 1))
        aux_loss = self.num_experts * torch.sum(importance * selected)
        return out.view(bsz, seq_len, dim), aux_loss

