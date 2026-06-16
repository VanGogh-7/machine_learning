import torch
from torch import nn


class WideDeepCTRModel(nn.Module):
    def __init__(
        self,
        num_numerical_features: int,
        category_sizes: list[int],
        embedding_dim: int,
        deep_hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_numerical_features = num_numerical_features
        self.num_categorical_features = len(category_sizes)

        # Wide categorical terms use one scalar per category to memorize sparse IDs.
        self.wide_embeddings = nn.ModuleList(
            nn.Embedding(category_size, 1)
            for category_size in category_sizes
        )
        self.wide_numerical = nn.Linear(num_numerical_features, 1)

        # Deep categorical embeddings turn sparse category IDs into dense vectors.
        self.deep_embeddings = nn.ModuleList(
            nn.Embedding(category_size, embedding_dim)
            for category_size in category_sizes
        )

        deep_input_dim = num_numerical_features + len(category_sizes) * embedding_dim
        deep_layers = []
        current_dim = deep_input_dim
        for hidden_dim in deep_hidden_dims:
            deep_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        deep_layers.append(nn.Linear(current_dim, 1))
        self.deep_network = nn.Sequential(*deep_layers)

    def forward(
        self,
        numerical_features: torch.Tensor,
        categorical_features: torch.Tensor,
    ) -> torch.Tensor:
        # numerical_features: [batch_size, num_numerical_features]
        # categorical_features: integer category IDs, [batch_size, num_fields]

        # The wide component handles memorization with linear numerical terms.
        wide_logits = self.wide_numerical(numerical_features).squeeze(1)
        for field_index, embedding in enumerate(self.wide_embeddings):
            field_ids = categorical_features[:, field_index]
            wide_logits = wide_logits + embedding(field_ids).squeeze(1)

        # The deep component handles generalization with embeddings and an MLP.
        deep_parts = [numerical_features]
        for field_index, embedding in enumerate(self.deep_embeddings):
            deep_parts.append(embedding(categorical_features[:, field_index]))
        deep_input = torch.cat(deep_parts, dim=1)
        deep_logits = self.deep_network(deep_input).squeeze(1)

        # Adding logits combines memorization and generalization before loss.
        # BCEWithLogitsLoss applies sigmoid internally, so forward returns raw logits.
        return wide_logits + deep_logits


if __name__ == "__main__":
    numerical = torch.randn(4, 13)
    categorical = torch.randint(0, 100, (4, 26))
    category_sizes = [100] * 26
    model = WideDeepCTRModel(
        num_numerical_features=13,
        category_sizes=category_sizes,
        embedding_dim=16,
        deep_hidden_dims=(128, 64, 32),
        dropout=0.2,
    )
    logits = model(numerical, categorical)
    print(logits.shape)
