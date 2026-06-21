from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .attention import CausalSelfAttention, KVCache
from .config import LLMConfig
from .ffn import SwiGLUFeedForward
from .moe import MoEFeedForward
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.use_moe = config.use_moe
        self.ffn = (
            MoEFeedForward(config.d_model, config.num_experts, config.top_k, config.dropout)
            if config.use_moe
            else SwiGLUFeedForward(config.d_model, dropout=config.dropout)
        )

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None, use_cache: bool = False):
        attn_out, new_cache = self.attn(self.attn_norm(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
        else:
            ffn_out = self.ffn(self.ffn_norm(x))
            aux_loss = x.new_zeros(())
        x = x + ffn_out
        return x, new_cache, aux_loss


class ModernDecoderLM(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        assert config.vocab_size > 0, "Set config.vocab_size after building the tokenizer."
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[KVCache]] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor | List[KVCache] | None]:
        # input_ids: [batch, seq]
        bsz, seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_len or use_cache
        x = self.dropout(self.token_embedding(input_ids))

        new_caches: List[KVCache] = []
        aux_losses = []
        for layer_idx, block in enumerate(self.blocks):
            layer_cache = kv_cache[layer_idx] if use_cache and kv_cache is not None else None
            x, new_cache, aux_loss = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                assert new_cache is not None
                new_caches.append(new_cache)
            aux_losses.append(aux_loss)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        assert logits.shape == (bsz, seq_len, self.config.vocab_size)

        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            loss = ce_loss + self.config.moe_aux_loss_weight * aux_loss if self.config.use_moe else ce_loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss,
            "kv_cache": new_caches if use_cache else None,
        }

