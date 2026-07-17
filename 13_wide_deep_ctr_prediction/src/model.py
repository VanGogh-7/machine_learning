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
        self.wide_embeddings = nn.ModuleList(
            nn.Embedding(category_size, 1)
            for category_size in category_sizes
        )
        self.deep_embeddings = nn.ModuleList(
            nn.Embedding(category_size, embedding_dim)
            for category_size in category_sizes
        )
        self.wide_numerical = nn.Linear(num_numerical_features, 1)

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
        # The wide component memorizes linear numerical and categorical patterns.
        wide_logit = self.wide_numerical(numerical_features).squeeze(1)
        for field_index, embedding in enumerate(self.wide_embeddings):
            wide_logit = wide_logit + embedding(
                categorical_features[:, field_index]
            ).squeeze(1)

        # The deep component generalizes through embeddings and nonlinear layers.
        deep_parts = [numerical_features]
        for field_index, embedding in enumerate(self.deep_embeddings):
            deep_parts.append(embedding(categorical_features[:, field_index]))
        deep_input = torch.cat(deep_parts, dim=1)
        deep_logit = self.deep_network(deep_input).squeeze(1)

        # BCEWithLogitsLoss applies sigmoid, so the model returns raw added logits.
        return wide_logit + deep_logit


if __name__ == "__main__":
    numerical = torch.randn(4, 13)
    categorical = torch.randint(0, 100, (4, 26))
    model = WideDeepCTRModel(
        num_numerical_features=13,
        category_sizes=[100] * 26,
        embedding_dim=16,
        deep_hidden_dims=(128, 64, 32),
        dropout=0.2,
    )
    logits = model(numerical, categorical)
    print(logits.shape)
