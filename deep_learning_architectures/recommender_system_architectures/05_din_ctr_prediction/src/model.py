import torch
from torch import nn


class ActivationUnit(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        layers = []
        current_dim = embedding_dim * 4
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        target_item_embedding: torch.Tensor,
        history_item_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # target_item_embedding: [batch_size, embedding_dim]
        # history_item_embeddings: [batch_size, max_history_length, embedding_dim]
        sequence_length = history_item_embeddings.size(1)
        target_expanded = target_item_embedding.unsqueeze(1).expand(
            -1,
            sequence_length,
            -1,
        )

        # The local activation unit scores each historical item for this target.
        activation_input = torch.cat(
            [
                history_item_embeddings,
                target_expanded,
                history_item_embeddings - target_expanded,
                history_item_embeddings * target_expanded,
            ],
            dim=2,
        )
        attention_scores = self.network(activation_input).squeeze(2)

        # Masked softmax prevents padded history positions from receiving weight.
        masked_scores = attention_scores.masked_fill(history_mask <= 0, -1e9)
        attention_weights = torch.softmax(masked_scores, dim=1)
        attention_weights = attention_weights * history_mask
        normalizer = attention_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attention_weights = attention_weights / normalizer

        # User interest is target-aware, not a fixed pooled history vector.
        user_interest_vector = torch.sum(
            history_item_embeddings * attention_weights.unsqueeze(2),
            dim=1,
        )
        return attention_scores, attention_weights, user_interest_vector


class DINCTRModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        max_history_length: int,
        activation_hidden_dims: tuple[int, ...],
        mlp_hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_history_length = max_history_length

        # Item id 0 is reserved for padded historical behavior positions.
        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
            padding_idx=0,
        )
        self.activation_unit = ActivationUnit(
            embedding_dim=embedding_dim,
            hidden_dims=activation_hidden_dims,
            dropout=dropout,
        )

        mlp_layers = []
        current_dim = embedding_dim * 2
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
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        # target_item_ids: candidate item ids, [batch_size]
        # history_item_ids: historical behavior sequence, [batch_size, max_length]
        # history_mask: 1 for real history items and 0 for padding.
        target_item_embedding = self.item_embedding(target_item_ids)
        history_item_embeddings = self.item_embedding(history_item_ids)

        _, _, user_interest_vector = self.activation_unit(
            target_item_embedding,
            history_item_embeddings,
            history_mask,
        )

        # Prediction combines the target item and target-aware user interest.
        prediction_input = torch.cat(
            [user_interest_vector, target_item_embedding],
            dim=1,
        )
        logits = self.prediction_mlp(prediction_input).squeeze(1)

        # BCEWithLogitsLoss applies sigmoid internally, so forward returns logits.
        return logits


if __name__ == "__main__":
    target_item_ids = torch.randint(1, 1000, (4,))
    history_item_ids = torch.randint(0, 1000, (4, 20))
    history_mask = (history_item_ids != 0).float()

    model = DINCTRModel(
        num_items=1000,
        embedding_dim=32,
        max_history_length=20,
        activation_hidden_dims=(64, 32),
        mlp_hidden_dims=(128, 64, 32),
        dropout=0.2,
    )

    logits = model(target_item_ids, history_item_ids, history_mask)
    print(logits.shape)
