import torch
from torch import nn


class CrossLayer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.xavier_uniform_(self.weight.unsqueeze(0))

    def forward(self, x0: torch.Tensor, x_l: torch.Tensor) -> torch.Tensor:
        # Original DCN formula:
        # x_{l+1} = x_0 * (x_l w_l) + b_l + x_l
        xw = torch.sum(x_l * self.weight, dim=1, keepdim=True)
        return x0 * xw + self.bias + x_l


class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            CrossLayer(input_dim) for _ in range(num_layers)
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        # Each cross layer explicitly crosses the current representation with x0.
        x_l = x0
        for layer in self.layers:
            x_l = layer(x0, x_l)
        return x_l


class DCNCTRModel(nn.Module):
    def __init__(
        self,
        num_numerical_features: int,
        category_sizes: list[int],
        embedding_dim: int,
        cross_num_layers: int,
        deep_hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_numerical_features = num_numerical_features
        self.num_categorical_features = len(category_sizes)
        self.input_dim = (
            num_numerical_features + len(category_sizes) * embedding_dim
        )

        # Categorical feature IDs are embedded before concatenation with numbers.
        self.embeddings = nn.ModuleList(
            nn.Embedding(category_size, embedding_dim)
            for category_size in category_sizes
        )

        self.cross_network = CrossNetwork(self.input_dim, cross_num_layers)

        deep_layers = []
        current_dim = self.input_dim
        for hidden_dim in deep_hidden_dims:
            deep_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        self.deep_network = nn.Sequential(*deep_layers)
        deep_output_dim = (
            deep_hidden_dims[-1] if len(deep_hidden_dims) > 0 else self.input_dim
        )

        # The final layer combines explicit crosses and implicit deep interactions.
        self.final_linear = nn.Linear(self.input_dim + deep_output_dim, 1)

    def build_input(
        self,
        numerical_features: torch.Tensor,
        categorical_features: torch.Tensor,
    ) -> torch.Tensor:
        # numerical_features: [batch_size, num_numerical_features]
        # categorical_features: integer category IDs, [batch_size, num_fields]
        embedding_parts = []
        for field_index, embedding in enumerate(self.embeddings):
            embedding_parts.append(embedding(categorical_features[:, field_index]))

        # x0 is the original input vector used by every cross layer.
        return torch.cat([numerical_features, *embedding_parts], dim=1)

    def forward(
        self,
        numerical_features: torch.Tensor,
        categorical_features: torch.Tensor,
    ) -> torch.Tensor:
        x0 = self.build_input(numerical_features, categorical_features)

        # Cross Network learns bounded-degree explicit feature crossing.
        cross_output = self.cross_network(x0)

        # Deep Network learns nonlinear feature interactions implicitly.
        deep_output = self.deep_network(x0)

        # Concatenate both views before producing one raw CTR logit.
        combined = torch.cat([cross_output, deep_output], dim=1)
        logits = self.final_linear(combined).squeeze(1)

        # BCEWithLogitsLoss applies sigmoid internally, so forward returns logits.
        return logits


if __name__ == "__main__":
    numerical = torch.randn(4, 13)
    categorical = torch.randint(0, 100, (4, 26))
    category_sizes = [100] * 26
    model = DCNCTRModel(
        num_numerical_features=13,
        category_sizes=category_sizes,
        embedding_dim=16,
        cross_num_layers=3,
        deep_hidden_dims=(128, 64, 32),
        dropout=0.2,
    )
    logits = model(numerical, categorical)
    print(logits.shape)
