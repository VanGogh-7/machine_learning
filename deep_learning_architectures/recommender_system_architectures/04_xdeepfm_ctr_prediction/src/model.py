import torch
from torch import nn


class CompressedInteractionNetwork(nn.Module):
    def __init__(
        self,
        num_fields: int,
        layer_sizes: tuple[int, ...],
        split_half: bool = False,
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.layer_sizes = layer_sizes
        self.split_half = split_half
        self.conv_layers = nn.ModuleList()
        self.output_dims = []

        previous_field_dim = num_fields
        for layer_index, layer_size in enumerate(layer_sizes):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=previous_field_dim * num_fields,
                    out_channels=layer_size,
                    kernel_size=1,
                )
            )
            is_last_layer = layer_index == len(layer_sizes) - 1
            if split_half and not is_last_layer:
                next_field_dim = layer_size // 2
                direct_output_dim = layer_size - next_field_dim
            else:
                next_field_dim = layer_size
                direct_output_dim = layer_size
            self.output_dims.append(direct_output_dim)
            previous_field_dim = next_field_dim

        self.output_dim = sum(self.output_dims)

    def forward(self, field_embeddings: torch.Tensor) -> torch.Tensor:
        # field_embeddings: [batch_size, num_fields, embedding_dim]
        x0 = field_embeddings
        xk = field_embeddings
        pooled_outputs = []

        for layer_index, conv_layer in enumerate(self.conv_layers):
            batch_size, previous_field_dim, embedding_dim = xk.shape

            # CIN forms vector-wise feature interactions between x0 and xk.
            # This differs from FM's second-order formula and DCN's scalar cross.
            interactions = torch.einsum("bhd,bmd->bhmd", xk, x0)
            interactions = interactions.reshape(
                batch_size,
                previous_field_dim * self.num_fields,
                embedding_dim,
            )

            # Conv1d compresses the interaction field dimension at each layer.
            x_next = torch.relu(conv_layer(interactions))

            is_last_layer = layer_index == len(self.conv_layers) - 1
            if self.split_half and not is_last_layer:
                next_field_dim = self.layer_sizes[layer_index] // 2
                xk, direct_output = torch.split(
                    x_next,
                    [next_field_dim, self.layer_sizes[layer_index] - next_field_dim],
                    dim=1,
                )
            else:
                xk = x_next
                direct_output = x_next

            pooled_outputs.append(direct_output.sum(dim=2))

        return torch.cat(pooled_outputs, dim=1)


class XDeepFMCTRModel(nn.Module):
    def __init__(
        self,
        num_numerical_features: int,
        category_sizes: list[int],
        embedding_dim: int,
        cin_layer_sizes: tuple[int, ...],
        deep_hidden_dims: tuple[int, ...],
        dropout: float,
        cin_split_half: bool = False,
    ) -> None:
        super().__init__()
        self.num_numerical_features = num_numerical_features
        self.num_categorical_features = len(category_sizes)
        self.embedding_dim = embedding_dim

        # First-order categorical terms provide the linear memorization component.
        self.linear_embeddings = nn.ModuleList(
            nn.Embedding(category_size, 1)
            for category_size in category_sizes
        )
        self.linear_numerical = nn.Linear(num_numerical_features, 1)

        # Categorical feature IDs are mapped to field embeddings for CIN and DNN.
        self.embeddings = nn.ModuleList(
            nn.Embedding(category_size, embedding_dim)
            for category_size in category_sizes
        )

        self.cin = CompressedInteractionNetwork(
            num_fields=len(category_sizes),
            layer_sizes=cin_layer_sizes,
            split_half=cin_split_half,
        )
        self.cin_linear = nn.Linear(self.cin.output_dim, 1)

        deep_input_dim = len(category_sizes) * embedding_dim + num_numerical_features
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

    def build_field_embeddings(self, categorical_features: torch.Tensor) -> torch.Tensor:
        # categorical_features: integer category IDs, [batch_size, num_fields]
        field_embeddings = [
            embedding(categorical_features[:, field_index])
            for field_index, embedding in enumerate(self.embeddings)
        ]
        return torch.stack(field_embeddings, dim=1)

    def forward(
        self,
        numerical_features: torch.Tensor,
        categorical_features: torch.Tensor,
    ) -> torch.Tensor:
        # numerical_features: [batch_size, num_numerical_features]
        # categorical_features: [batch_size, num_categorical_features]
        field_embeddings = self.build_field_embeddings(categorical_features)

        linear_logit = self.linear_numerical(numerical_features).squeeze(1)
        for field_index, embedding in enumerate(self.linear_embeddings):
            linear_logit = linear_logit + embedding(
                categorical_features[:, field_index]
            ).squeeze(1)

        # CIN explicitly learns vector-wise high-order feature interactions.
        cin_output = self.cin(field_embeddings)
        cin_logit = self.cin_linear(cin_output).squeeze(1)

        # The deep component learns implicit nonlinear interactions.
        flattened_embeddings = field_embeddings.reshape(field_embeddings.size(0), -1)
        deep_input = torch.cat([flattened_embeddings, numerical_features], dim=1)
        deep_logit = self.deep_network(deep_input).squeeze(1)

        # xDeepFM adds linear, explicit CIN, and implicit deep logits.
        # BCEWithLogitsLoss applies sigmoid internally, so forward returns logits.
        return linear_logit + cin_logit + deep_logit


if __name__ == "__main__":
    numerical = torch.randn(4, 13)
    categorical = torch.randint(0, 100, (4, 26))
    category_sizes = [100] * 26
    model = XDeepFMCTRModel(
        num_numerical_features=13,
        category_sizes=category_sizes,
        embedding_dim=16,
        cin_layer_sizes=(64, 64, 64),
        deep_hidden_dims=(128, 64, 32),
        dropout=0.2,
    )
    logits = model(numerical, categorical)
    print(logits.shape)
