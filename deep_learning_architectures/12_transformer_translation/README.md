# Transformer Translation

This PyTorch project implements a small Transformer encoder-decoder for
English-to-German translation using the real Multi30k dataset.

## Transformer

Transformer replaced recurrent sequence modeling with attention-based sequence
modeling. Unlike RNN, LSTM, GRU, and classic Seq2Seq models, it processes token
relationships in parallel. This made Transformer historically important for
machine translation and later large language models.

This project is a direct comparison with `10_seq2seq_attention_translation`.
That project uses a recurrent encoder-decoder with attention. This project
removes recurrence and uses:

- Source and target token embeddings.
- Positional encoding to represent token order.
- Encoder self-attention over source tokens.
- Decoder masked self-attention over generated target tokens.
- Encoder-decoder cross-attention over encoder memory.
- Multi-head attention to learn several attention patterns.
- Feed-forward networks, residual connections, and layer normalization inside
  PyTorch's Transformer layers.
- Padding masks to ignore `<pad>` positions.
- A target causal mask to prevent attention to future target tokens.
- An output projection from decoder representations to target vocabulary
  logits.

## Dataset

The project reuses the shared Multi30k English-German dataset:

```text
machine_learning/datasets/multi30k/
```

Expected files:

```text
train.en
train.de
valid.en
valid.de
test.en
test.de
```

English is the source language and German is the target language. Source and
target vocabularies are built from the training split only. The validation
split selects the best checkpoint by validation loss. The test split is
evaluated only once after training.

## Tensor Shapes And Target Shifting

- Source input: `[batch_size, src_seq_len]`
- Target input: `[batch_size, tgt_seq_len]`
- Output logits: `[batch_size, tgt_seq_len, target_vocab_size]`

During training:

```text
decoder input  = target[:, :-1]
decoder target = target[:, 1:]
```

For `<bos> ich mag dieses bild <eos>`, the decoder receives
`<bos> ich mag dieses bild` and learns to predict `ich mag dieses bild <eos>`.

## Project Structure

```text
12_transformer_translation/
├── scripts/
│   ├── train.py
│   └── predict.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   ├── model.py
│   ├── utils.py
│   └── visualize.py
├── README.md
└── requirements.txt
```

## Train

```bash
python scripts/train.py
```

Training saves the best validation-loss checkpoint, both vocabularies, and
training history.

## Predict

After training:

```bash
python scripts/predict.py
```

This is a small educational Transformer translation model, not a pretrained
translation system.

## Model Summary and Key Papers

### Historical Background

The Transformer was introduced for machine translation and replaced recurrent encoder-decoder models with attention-based sequence modeling.

### Basic Structure

The architecture uses encoder self-attention, decoder masked self-attention, encoder-decoder cross-attention, feed-forward layers, residual connections, and layer normalization.

### Why It Matters

Transformer encoder-decoder models became a foundation for modern machine translation, summarization, and many sequence-to-sequence systems.

### Key Papers

* [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
