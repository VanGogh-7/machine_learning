import torch
from torch import nn
from torch.nn import functional as F


def build_tower_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class UserTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # The user tower maps user ids into retrieval embeddings.
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp = build_tower_mlp(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        user_embedding = self.user_embedding(user_ids)
        user_vector = self.mlp(user_embedding)
        return F.normalize(user_vector, p=2, dim=1)


class ItemTower(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # The item tower maps item ids into retrieval embeddings.
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = build_tower_mlp(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_embedding = self.item_embedding(item_ids)
        item_vector = self.mlp(item_embedding)
        return F.normalize(item_vector, p=2, dim=1)


class TwoTowerRetrievalModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_embedding_dim: int,
        item_embedding_dim: int,
        tower_hidden_dims: tuple[int, ...],
        output_dim: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.user_tower = UserTower(
            num_users=num_users,
            embedding_dim=user_embedding_dim,
            hidden_dims=tower_hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.item_tower = ItemTower(
            num_items=num_items,
            embedding_dim=item_embedding_dim,
            hidden_dims=tower_hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.temperature = temperature
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_ids)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        # Item embeddings can be precomputed offline for fast candidate retrieval.
        return self.item_tower(item_ids)

    def score_pairs(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_vectors = self.encode_users(user_ids)
        item_vectors = self.encode_items(item_ids)
        return (user_vectors * item_vectors).sum(dim=1) / self.temperature

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        user_vectors = self.encode_users(user_ids)
        item_vectors = self.encode_items(item_ids)
        # In-batch negatives use every other item in the batch as a negative.
        # scores[i, j] is the similarity between user i and item j in the batch.
        scores = user_vectors @ item_vectors.T
        # Temperature scaling sharpens or softens the retrieval distribution.
        scores = scores / self.temperature
        # Separate towers make this suitable for candidate retrieval because item
        # vectors can be computed once and searched with dot product similarity.
        return {
            "user_vectors": user_vectors,
            "item_vectors": item_vectors,
            "scores": scores,
        }


if __name__ == "__main__":
    user_ids = torch.randint(0, 1000, (4,))
    item_ids = torch.randint(0, 2000, (4,))

    model = TwoTowerRetrievalModel(
        num_users=1000,
        num_items=2000,
        user_embedding_dim=64,
        item_embedding_dim=64,
        tower_hidden_dims=(128, 64),
        output_dim=64,
        dropout=0.2,
        temperature=0.07,
    )
    outputs = model(user_ids, item_ids)
    print(outputs["user_vectors"].shape)
    print(outputs["item_vectors"].shape)
    print(outputs["scores"].shape)
