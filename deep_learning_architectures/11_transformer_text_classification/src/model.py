import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, embedding_dim, 2)
            * (-math.log(10000.0) / embedding_dim)
        )
        encoding = torch.zeros(max_length, embedding_dim)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        sequence_length = embedded.size(1)
        return self.dropout(embedded + self.encoding[:, :sequence_length])


class TransformerTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_encoder_layers: int,
        dropout: float,
        n_classes: int,
        pad_idx: int,
        max_length: int,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.pad_idx = pad_idx
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx,
        )
        self.positional_encoding = PositionalEncoding(
            embedding_dim,
            max_length,
            dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Token IDs become dense vectors, then positional encoding adds token order.
        embedded = self.embedding(input_ids) * math.sqrt(self.embedding_dim)
        encoded_input = self.positional_encoding(embedded)

        # Self-attention uses this mask to ignore padding token positions.
        padding_mask = input_ids == self.pad_idx
        encoder_output = self.encoder(
            encoded_input,
            src_key_padding_mask=padding_mask,
        )

        # Mean pooling summarizes only non-padding Transformer outputs.
        non_padding_mask = (~padding_mask).unsqueeze(-1)
        pooled_output = (encoder_output * non_padding_mask).sum(dim=1)
        token_counts = non_padding_mask.sum(dim=1).clamp(min=1)
        pooled_output = pooled_output / token_counts
        return self.classifier(self.dropout(pooled_output))


if __name__ == "__main__":
    x = torch.randint(0, 100, (4, 20))
    model = TransformerTextClassifier(
        vocab_size=100,
        embedding_dim=128,
        num_heads=4,
        feedforward_dim=256,
        num_encoder_layers=2,
        dropout=0.1,
        n_classes=2,
        pad_idx=0,
        max_length=20,
    )
    y = model(x)
    print(y.shape)
