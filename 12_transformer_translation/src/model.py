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

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        sequence_length = embeddings.size(1)
        return self.dropout(embeddings + self.encoding[:, :sequence_length])


class TransformerTranslationModel(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
        src_pad_idx: int,
        tgt_pad_idx: int,
        max_length: int,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.embedding_dim = embedding_dim
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.src_embedding = nn.Embedding(
            src_vocab_size,
            embedding_dim,
            padding_idx=src_pad_idx,
        )
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            embedding_dim,
            padding_idx=tgt_pad_idx,
        )
        self.positional_encoding = PositionalEncoding(
            embedding_dim,
            max_length,
            dropout,
        )
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(embedding_dim, tgt_vocab_size)

    def make_src_padding_mask(self, src_input_ids: torch.Tensor) -> torch.Tensor:
        return src_input_ids == self.src_pad_idx

    def make_tgt_padding_mask(self, tgt_input_ids: torch.Tensor) -> torch.Tensor:
        return tgt_input_ids == self.tgt_pad_idx

    @staticmethod
    def generate_square_subsequent_mask(
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    def encode(
        self,
        src_input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_padding_mask = self.make_src_padding_mask(src_input_ids)
        src_embeddings = self.src_embedding(src_input_ids)
        src_embeddings = src_embeddings * math.sqrt(self.embedding_dim)
        src_embeddings = self.positional_encoding(src_embeddings)
        memory = self.transformer.encoder(
            src_embeddings,
            src_key_padding_mask=src_padding_mask,
        )
        return memory, src_padding_mask

    def decode(
        self,
        tgt_input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_padding_mask = self.make_tgt_padding_mask(tgt_input_ids)
        tgt_causal_mask = self.generate_square_subsequent_mask(
            tgt_input_ids.size(1),
            tgt_input_ids.device,
        )
        tgt_embeddings = self.tgt_embedding(tgt_input_ids)
        tgt_embeddings = tgt_embeddings * math.sqrt(self.embedding_dim)
        tgt_embeddings = self.positional_encoding(tgt_embeddings)
        return self.transformer.decoder(
            tgt=tgt_embeddings,
            memory=memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

    def forward(
        self,
        src_input_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        # The encoder uses source self-attention with a source padding mask.
        memory, src_padding_mask = self.encode(src_input_ids)

        # The decoder uses masked self-attention and cross-attends to encoder memory.
        decoder_output = self.decode(
            tgt_input_ids,
            memory,
            src_padding_mask,
        )
        return self.output_projection(decoder_output)

    def generate(
        self,
        src_input_ids: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_length: int,
    ) -> list[int]:
        if src_input_ids.size(0) != 1:
            raise ValueError("generate expects one source sentence.")

        self.eval()
        with torch.no_grad():
            memory, src_padding_mask = self.encode(src_input_ids)
            generated = torch.tensor(
                [[bos_idx]],
                dtype=torch.long,
                device=src_input_ids.device,
            )
            for _ in range(max_length - 1):
                decoder_output = self.decode(
                    generated,
                    memory,
                    src_padding_mask,
                )
                next_token = self.output_projection(decoder_output[:, -1]).argmax(
                    dim=1,
                    keepdim=True,
                )
                generated = torch.cat((generated, next_token), dim=1)
                if next_token.item() == eos_idx:
                    break
        return generated.squeeze(0).tolist()


if __name__ == "__main__":
    src = torch.randint(1, 100, (4, 20))
    tgt = torch.randint(1, 120, (4, 18))
    model = TransformerTranslationModel(
        src_vocab_size=100,
        tgt_vocab_size=120,
        embedding_dim=128,
        num_heads=4,
        feedforward_dim=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        src_pad_idx=0,
        tgt_pad_idx=0,
        max_length=20,
    )
    logits = model(src, tgt)
    print(logits.shape)
