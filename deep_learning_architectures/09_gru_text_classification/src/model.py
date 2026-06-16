import torch
from torch import nn


class GRUTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        n_classes: int,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx,
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        num_directions = 2 if bidirectional else 1
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * num_directions, n_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Token ids become dense embedding vectors for each sequence position.
        embedded = self.embedding(input_ids)
        lengths = (input_ids != self.pad_idx).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        # Update and reset gates control the GRU hidden state without a cell state.
        _, final_hidden = self.gru(packed)

        if self.bidirectional:
            # Concatenate the last layer's forward and backward hidden states.
            sequence_state = torch.cat((final_hidden[-2], final_hidden[-1]), dim=1)
        else:
            # The last layer's final hidden state summarizes the sequence.
            sequence_state = final_hidden[-1]

        return self.classifier(self.output_dropout(sequence_state))


if __name__ == "__main__":
    x = torch.randint(0, 100, (4, 20))
    model = GRUTextClassifier(
        vocab_size=100,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        bidirectional=False,
        dropout=0.0,
        n_classes=2,
        pad_idx=0,
    )
    y = model(x)
    print(y.shape)
