import torch
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, gru_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.target_projection = nn.Linear(embedding_dim, gru_hidden_dim)
        self.scoring_network = nn.Sequential(
            nn.Linear(gru_hidden_dim * 4, gru_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim, 1),
        )

    def forward(
        self,
        target_item_embedding: torch.Tensor,
        interest_states: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Target-aware attention scores every extracted interest state for the
        # current candidate item.
        sequence_length = interest_states.size(1)
        target_state = self.target_projection(target_item_embedding)
        target_expanded = target_state.unsqueeze(1).expand(-1, sequence_length, -1)
        attention_input = torch.cat(
            [
                interest_states,
                target_expanded,
                interest_states - target_expanded,
                interest_states * target_expanded,
            ],
            dim=2,
        )
        attention_scores = self.scoring_network(attention_input).squeeze(2)
        masked_scores = attention_scores.masked_fill(history_mask <= 0, -1e9)
        attention_weights = torch.softmax(masked_scores, dim=1)
        attention_weights = attention_weights * history_mask
        normalizer = attention_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attention_weights = attention_weights / normalizer
        return attention_scores, attention_weights


class AUGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_gates = nn.Linear(input_dim, hidden_dim * 3)
        self.hidden_gates = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)

    def forward(
        self,
        input_t: torch.Tensor,
        hidden_prev: torch.Tensor,
        attention_t: torch.Tensor,
    ) -> torch.Tensor:
        input_reset, input_update, input_candidate = self.input_gates(input_t).chunk(
            3,
            dim=1,
        )
        hidden_reset, hidden_update, hidden_candidate = self.hidden_gates(
            hidden_prev
        ).chunk(3, dim=1)

        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        candidate_hidden = torch.tanh(
            input_candidate + reset_gate * hidden_candidate
        )

        # AUGRU scales the update gate by target-aware attention.
        attention_update_gate = update_gate * attention_t.unsqueeze(1)
        return (
            (1.0 - attention_update_gate) * hidden_prev
            + attention_update_gate * candidate_hidden
        )


class AUGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = AUGRUCell(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_weights: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        hidden = inputs.new_zeros(batch_size, self.hidden_dim)

        for step in range(sequence_length):
            next_hidden = self.cell(
                input_t=inputs[:, step, :],
                hidden_prev=hidden,
                attention_t=attention_weights[:, step],
            )
            mask_t = history_mask[:, step].unsqueeze(1)
            hidden = mask_t * next_hidden + (1.0 - mask_t) * hidden

        return hidden


class DIENCTRModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        gru_hidden_dim: int,
        max_history_length: int,
        mlp_hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.max_history_length = max_history_length

        # Item id 0 is reserved for padded historical behavior positions.
        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
            padding_idx=0,
        )

        # The interest extraction GRU turns item behavior embeddings into latent
        # interest states at each sequence step.
        self.interest_extraction_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )

        # Auxiliary next-item scoring is an educational approximation of DIEN's
        # original auxiliary loss. It scores the observed next behavior only.
        self.aux_projection = nn.Linear(gru_hidden_dim, embedding_dim)
        self.attention_layer = AttentionLayer(
            embedding_dim=embedding_dim,
            gru_hidden_dim=gru_hidden_dim,
            dropout=dropout,
        )
        self.interest_evolution_augru = AUGRU(
            input_dim=gru_hidden_dim,
            hidden_dim=gru_hidden_dim,
        )

        mlp_layers = []
        current_dim = gru_hidden_dim + embedding_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        mlp_layers.append(nn.Linear(current_dim, 1))
        self.prediction_mlp = nn.Sequential(*mlp_layers)

    def forward(
        self,
        target_item_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        next_history_item_ids: torch.Tensor,
        history_mask: torch.Tensor,
        aux_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # target_item_ids: candidate item ids, [batch_size]
        # history_item_ids: user behavior sequence, [batch_size, max_history_length]
        # next_history_item_ids: shifted next behavior ids for auxiliary loss.
        target_item_embedding = self.item_embedding(target_item_ids)
        history_item_embeddings = self.item_embedding(history_item_ids)
        next_item_embeddings = self.item_embedding(next_history_item_ids)

        interest_states, _ = self.interest_extraction_gru(history_item_embeddings)

        projected_interest_states = self.aux_projection(interest_states)
        aux_logits = torch.sum(
            projected_interest_states * next_item_embeddings,
            dim=2,
        )

        _, attention_weights = self.attention_layer(
            target_item_embedding=target_item_embedding,
            interest_states=interest_states,
            history_mask=history_mask,
        )
        evolved_interest = self.interest_evolution_augru(
            inputs=interest_states,
            attention_weights=attention_weights,
            history_mask=history_mask,
        )

        prediction_input = torch.cat(
            [evolved_interest, target_item_embedding],
            dim=1,
        )
        logits = self.prediction_mlp(prediction_input).squeeze(1)

        # BCEWithLogitsLoss applies sigmoid internally, so forward returns raw
        # logits. Sigmoid is used only for metrics and prediction output.
        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "aux_mask": aux_mask,
            "attention_weights": attention_weights,
        }


if __name__ == "__main__":
    target_item_ids = torch.randint(1, 1000, (4,))
    history_item_ids = torch.randint(0, 1000, (4, 20))
    next_history_item_ids = torch.randint(0, 1000, (4, 20))
    history_mask = (history_item_ids != 0).float()
    aux_mask = (next_history_item_ids != 0).float()

    model = DIENCTRModel(
        num_items=1000,
        embedding_dim=32,
        gru_hidden_dim=64,
        max_history_length=20,
        mlp_hidden_dims=(128, 64, 32),
        dropout=0.2,
    )

    outputs = model(
        target_item_ids=target_item_ids,
        history_item_ids=history_item_ids,
        next_history_item_ids=next_history_item_ids,
        history_mask=history_mask,
        aux_mask=aux_mask,
    )

    print(outputs["logits"].shape)
    print(outputs["aux_logits"].shape)
    print(outputs["attention_weights"].shape)
