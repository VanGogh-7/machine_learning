import torch
from torch import nn


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        # Initial embeddings are the only trainable parameters in LightGCN.
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

    def computer(self, normalized_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Concatenate user and item graph nodes before sparse propagation.
        all_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0,
        )
        layer_embeddings = [all_embeddings]

        # LightGCN keeps only sparse normalized neighborhood aggregation.
        # It removes feature transformation matrices and nonlinear activations.
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(normalized_adj, all_embeddings)
            layer_embeddings.append(all_embeddings)

        # Average layer 0 through layer K embeddings.
        final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        final_user_embeddings, final_item_embeddings = torch.split(
            final_embeddings,
            [self.num_users, self.num_items],
            dim=0,
        )
        return final_user_embeddings, final_item_embeddings

    def get_initial_embeddings(
        self,
        user_ids: torch.Tensor,
        positive_item_ids: torch.Tensor,
        negative_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.user_embedding(user_ids),
            self.item_embedding(positive_item_ids),
            self.item_embedding(negative_item_ids),
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        positive_item_ids: torch.Tensor,
        negative_item_ids: torch.Tensor,
        normalized_adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final_user_embeddings, final_item_embeddings = self.computer(normalized_adj)
        user_vectors = final_user_embeddings[user_ids]
        positive_item_vectors = final_item_embeddings[positive_item_ids]
        negative_item_vectors = final_item_embeddings[negative_item_ids]
        # BPR compares a positive implicit-feedback edge against a sampled negative.
        positive_scores = torch.sum(user_vectors * positive_item_vectors, dim=1)
        negative_scores = torch.sum(user_vectors * negative_item_vectors, dim=1)
        return positive_scores, negative_scores

    def score_items(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        normalized_adj: torch.Tensor,
    ) -> torch.Tensor:
        final_user_embeddings, final_item_embeddings = self.computer(normalized_adj)
        user_vectors = final_user_embeddings[user_ids]
        item_vectors = final_item_embeddings[item_ids]
        return torch.sum(user_vectors * item_vectors, dim=1)

    def full_sort_scores(
        self,
        user_ids: torch.Tensor,
        normalized_adj: torch.Tensor,
    ) -> torch.Tensor:
        final_user_embeddings, final_item_embeddings = self.computer(normalized_adj)
        user_vectors = final_user_embeddings[user_ids]
        return user_vectors @ final_item_embeddings.T


if __name__ == "__main__":
    num_users = 3
    num_items = 4
    embedding_dim = 8
    num_layers = 2
    num_nodes = num_users + num_items
    edge_pairs = [(0, 3), (3, 0), (1, 5), (5, 1), (2, 6), (6, 2)]
    indices = torch.tensor(edge_pairs, dtype=torch.long).T
    values = torch.ones(indices.size(1))
    normalized_adj = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_nodes, num_nodes),
    ).coalesce()

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    )
    user_ids = torch.tensor([0, 1])
    positive_item_ids = torch.tensor([1, 2])
    negative_item_ids = torch.tensor([3, 0])
    positive_scores, negative_scores = model(
        user_ids,
        positive_item_ids,
        negative_item_ids,
        normalized_adj,
    )
    print(positive_scores.shape)
    print(negative_scores.shape)
