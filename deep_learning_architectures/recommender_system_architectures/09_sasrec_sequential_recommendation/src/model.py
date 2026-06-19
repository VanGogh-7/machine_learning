import torch
from torch import nn


class SASRecBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
        )
        self.feedforward_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Causal self-attention prevents each position from seeing future items.
        attention_output, _ = self.self_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden_states = self.attention_norm(
            hidden_states + self.dropout(attention_output)
        )
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        feedforward_output = self.feedforward(hidden_states)
        hidden_states = self.feedforward_norm(
            hidden_states + self.dropout(feedforward_output)
        )
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return hidden_states


class SASRecModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_attention_heads: int,
        num_transformer_blocks: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        # Item id 0 is reserved for padding.
        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
            padding_idx=0,
        )
        # Positional embeddings let self-attention know item order.
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                SASRecBlock(
                    embedding_dim=embedding_dim,
                    num_attention_heads=num_attention_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                )
                for _ in range(num_transformer_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _causal_mask(self, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(
                self.max_sequence_length,
                self.max_sequence_length,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    def encode_sequence(self, input_item_ids: torch.Tensor) -> torch.Tensor:
        # Sequential recommendation models ordered user behavior histories.
        padding_mask = input_item_ids == 0
        positions = torch.arange(
            self.max_sequence_length,
            device=input_item_ids.device,
        ).unsqueeze(0)
        item_embeddings = self.item_embedding(input_item_ids)
        position_embeddings = self.position_embedding(positions)
        hidden_states = self.embedding_dropout(item_embeddings + position_embeddings)
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        causal_mask = self._causal_mask(input_item_ids.device)
        for block in self.blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                causal_mask=causal_mask,
                padding_mask=padding_mask,
            )
        hidden_states = self.output_norm(hidden_states)

        # Use the final non-padding position as the sequence representation.
        # The prepared sequences are left-padded, so the final valid item is not
        # at sequence_length - 1. Compute its actual position explicitly.
        valid_positions = input_item_ids != 0
        position_indices = torch.arange(
            self.max_sequence_length,
            device=input_item_ids.device,
        ).unsqueeze(0)
        last_indices = (position_indices * valid_positions.long()).max(dim=1).values
        batch_indices = torch.arange(input_item_ids.size(0), device=input_item_ids.device)
        sequence_vector = hidden_states[batch_indices, last_indices]
        all_padding = valid_positions.sum(dim=1) == 0
        if all_padding.any():
            sequence_vector = sequence_vector.masked_fill(all_padding.unsqueeze(1), 0.0)
        return sequence_vector

    def score_items(
        self,
        input_item_ids: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        sequence_vector = self.encode_sequence(input_item_ids)
        target_embedding = self.item_embedding(target_item_ids)
        # Dot product scores whether the target is a likely next item.
        return torch.sum(sequence_vector * target_embedding, dim=1)

    def recommend_top_k(
        self,
        input_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_vector = self.encode_sequence(input_item_ids)
        candidate_embeddings = self.item_embedding(candidate_item_ids)
        scores = sequence_vector @ candidate_embeddings.T
        effective_top_k = min(top_k, scores.size(1))
        top_scores, top_positions = scores.topk(effective_top_k, dim=1)
        top_item_ids = candidate_item_ids[top_positions]
        return top_item_ids, top_scores

    def forward(
        self,
        input_item_ids: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.score_items(input_item_ids, target_item_ids)
        # BCEWithLogitsLoss applies sigmoid internally for numerical stability.
        return logits


if __name__ == "__main__":
    input_item_ids = torch.randint(0, 1000, (4, 50))
    target_item_ids = torch.randint(1, 1000, (4,))

    model = SASRecModel(
        num_items=1000,
        max_sequence_length=50,
        embedding_dim=64,
        num_attention_heads=2,
        num_transformer_blocks=2,
        feedforward_dim=128,
        dropout=0.2,
    )
    logits = model(input_item_ids, target_item_ids)
    print(logits.shape)
