import torch
from torch import nn


class DeepFMCTRModel(nn.Module):
    def __init__(
        self,
        num_numerical_features: int,
        category_sizes: list[int],
        embedding_dim: int,
        deep_hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.first_order_embeddings = nn.ModuleList(
            nn.Embedding(category_size, 1)
            for category_size in category_sizes
        )
        self.feature_embeddings = nn.ModuleList(
            nn.Embedding(category_size, embedding_dim)
            for category_size in category_sizes
        )
        self.first_order_numerical = nn.Linear(num_numerical_features, 1)

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
        # First-order FM terms model individual numerical and categorical effects.
        first_order_logit = self.first_order_numerical(numerical_features).squeeze(1)
        for field_index, embedding in enumerate(self.first_order_embeddings):
            first_order_logit = first_order_logit + embedding(
                categorical_features[:, field_index]
            ).squeeze(1)

        # Shared categorical embeddings support FM interactions and the deep part.
        field_embeddings = torch.stack(
            [
                embedding(categorical_features[:, field_index])
                for field_index, embedding in enumerate(self.feature_embeddings)
            ],
            dim=1,
        )
        square_of_sum = field_embeddings.sum(dim=1) ** 2
        sum_of_square = (field_embeddings ** 2).sum(dim=1)
        second_order_logit = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)

        # The deep component learns higher-order nonlinear feature interactions.
        flattened_embeddings = field_embeddings.flatten(start_dim=1)
        deep_input = torch.cat((numerical_features, flattened_embeddings), dim=1)
        deep_logit = self.deep_network(deep_input).squeeze(1)

        # BCEWithLogitsLoss applies sigmoid, so DeepFM returns raw added logits.
        return first_order_logit + second_order_logit + deep_logit


if __name__ == "__main__":
    numerical = torch.randn(4, 13)
    categorical = torch.randint(0, 100, (4, 26))
    model = DeepFMCTRModel(
        num_numerical_features=13,
        category_sizes=[100] * 26,
        embedding_dim=16,
        deep_hidden_dims=(128, 64, 32),
        dropout=0.2,
    )
    logits = model(numerical, categorical)
    print(logits.shape)
