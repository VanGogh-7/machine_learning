import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, src_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src_ids))
        lengths = (src_ids != self.pad_idx).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=src_ids.size(1),
        )
        return outputs, hidden


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.encoder_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_projection = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.decoder_projection(decoder_hidden).unsqueeze(1)
        keys = self.encoder_projection(encoder_outputs)
        scores = self.score_projection(torch.tanh(keys + query)).squeeze(-1)
        scores = scores.masked_fill(~src_mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.attention = AdditiveAttention(hidden_dim)
        self.gru = nn.GRU(
            embedding_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(embedding_dim + hidden_dim * 2, vocab_size)

    def forward(
        self,
        input_token: torch.Tensor,
        previous_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_token))
        context, attention_weights = self.attention(
            previous_hidden[-1],
            encoder_outputs,
            src_mask,
        )
        gru_input = torch.cat((embedded, context), dim=1).unsqueeze(1)
        decoder_output, hidden = self.gru(gru_input, previous_hidden)
        logits = self.output(
            torch.cat(
                (decoder_output.squeeze(1), context, embedded),
                dim=1,
            )
        )
        return logits, hidden, attention_weights


class Seq2SeqAttention(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_pad_idx: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        encoder_outputs, hidden = self.encoder(src_ids)
        src_mask = src_ids != self.src_pad_idx
        input_token = tgt_input_ids[:, 0]
        step_logits = []

        for step in range(tgt_input_ids.size(1)):
            logits, hidden, _ = self.decoder(
                input_token,
                hidden,
                encoder_outputs,
                src_mask,
            )
            step_logits.append(logits)
            if step + 1 < tgt_input_ids.size(1):
                predicted_token = logits.argmax(dim=1)
                use_teacher = torch.rand(1).item() < teacher_forcing_ratio
                input_token = (
                    tgt_input_ids[:, step + 1]
                    if use_teacher
                    else predicted_token
                )

        return torch.stack(step_logits, dim=1)

    def generate(
        self,
        src_ids: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_length: int,
    ) -> list[int]:
        if src_ids.size(0) != 1:
            raise ValueError("generate expects a batch containing one source sentence.")

        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src_ids)
            src_mask = src_ids != self.src_pad_idx
            input_token = torch.tensor([bos_idx], device=src_ids.device)
            generated_ids = []

            for _ in range(max_length):
                logits, hidden, _ = self.decoder(
                    input_token,
                    hidden,
                    encoder_outputs,
                    src_mask,
                )
                input_token = logits.argmax(dim=1)
                token_id = input_token.item()
                generated_ids.append(token_id)
                if token_id == eos_idx:
                    break

        return generated_ids


if __name__ == "__main__":
    encoder = Encoder(100, 32, 64, 1, 0.1, 0)
    decoder = Decoder(120, 32, 64, 1, 0.1, 0)
    model = Seq2SeqAttention(encoder, decoder, src_pad_idx=0)
    src = torch.randint(1, 100, (4, 12))
    tgt = torch.randint(1, 120, (4, 10))
    print(model(src, tgt).shape)
