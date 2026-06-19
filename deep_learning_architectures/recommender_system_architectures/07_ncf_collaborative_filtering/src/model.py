import torch
from torch import nn


class GMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        # User id embedding maps each encoded user to a dense trainable vector.
        self.user_gmf_embedding = nn.Embedding(num_users, embedding_dim)
        # Item id embedding maps each encoded item to a dense trainable vector.
        self.item_gmf_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_embedding = self.user_gmf_embedding(user_ids)
        item_embedding = self.item_gmf_embedding(item_ids)
        # Matrix factorization scores interactions through user-item embedding overlap.
        # GMF keeps that idea but learns embeddings for an elementwise product vector.
        return user_embedding * item_embedding


class MLPInteraction(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        # The MLP branch uses separate user and item embeddings from the GMF branch.
        self.user_mlp_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_mlp_embedding = nn.Embedding(num_items, embedding_dim)

        layers: list[nn.Module] = []
        input_dim = 2 * embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = input_dim

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_embedding = self.user_mlp_embedding(user_ids)
        item_embedding = self.item_mlp_embedding(item_ids)
        # MLP interaction modeling lets the network learn nonlinear user-item patterns.
        mlp_input = torch.cat([user_embedding, item_embedding], dim=1)
        return self.mlp(mlp_input)


class NeuMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_embedding_dim: int,
        mlp_embedding_dim: int,
        mlp_hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.gmf = GMF(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=gmf_embedding_dim,
        )
        self.mlp_interaction = MLPInteraction(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=mlp_embedding_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=dropout,
        )
        fusion_dim = gmf_embedding_dim + self.mlp_interaction.output_dim
        # NeuMF fusion combines the linear GMF signal and nonlinear MLP signal.
        self.output_layer = nn.Linear(fusion_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # In implicit feedback, the model predicts whether an interaction is likely.
        gmf_vector = self.gmf(user_ids, item_ids)
        mlp_vector = self.mlp_interaction(user_ids, item_ids)
        fusion_input = torch.cat([gmf_vector, mlp_vector], dim=1)
        logits = self.output_layer(fusion_input).squeeze(1)
        # BCEWithLogitsLoss applies sigmoid internally for better numerical stability.
        return logits


if __name__ == "__main__":
    user_ids = torch.randint(0, 1000, (4,))
    item_ids = torch.randint(0, 2000, (4,))

    model = NeuMF(
        num_users=1000,
        num_items=2000,
        gmf_embedding_dim=32,
        mlp_embedding_dim=32,
        mlp_hidden_dims=(128, 64, 32),
        dropout=0.2,
    )
    logits = model(user_ids, item_ids)
    print(logits.shape)
