import torch
from torch import nn


class RNNTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        n_classes: int,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx,
        )
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Token ids become dense embedding vectors for each sequence position.
        embedded = self.embedding(input_ids)
        # The RNN updates a hidden state while reading embeddings in order.
        _, final_hidden = self.rnn(embedded)
        # The final layer's hidden state summarizes the sequence for classification.
        return self.classifier(final_hidden[-1])


if __name__ == "__main__":
    x = torch.randint(0, 100, (4, 20))
    model = RNNTextClassifier(
        vocab_size=100,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        n_classes=2,
        pad_idx=0,
    )
    y = model(x)
    print(y.shape)
