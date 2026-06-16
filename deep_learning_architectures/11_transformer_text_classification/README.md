# Transformer Text Classification

This PyTorch project implements a small Transformer Encoder for sentiment
classification without pretrained embeddings or language models.

## Transformer

Transformer replaced recurrent sequence processing with self-attention. Unlike
RNN, LSTM, and GRU models, it can model relationships between all tokens in
parallel. It built on the attention mechanism used by Seq2Seq models and became
the foundation of modern NLP systems.

This educational classifier uses:

- **Token embeddings** to map token IDs to dense vectors.
- **Positional encoding** to represent token order without recurrence.
- **Self-attention** to connect every token with other tokens in the sequence.
- **Multi-head attention** to learn several attention patterns in parallel.
- **Feed-forward networks** to transform each token representation.
- **Residual connections and layer normalization** inside PyTorch's Transformer
  encoder layers.
- **A padding mask** to prevent attention from using `<pad>` positions.
- **Mean pooling** over non-padding encoder outputs for classification.

## Dataset

The project reuses the shared small sentiment classification dataset:

```text
machine_learning/datasets/text_classification/simple_sentiment.csv
```

It contains `text` and `label` columns, where `0` is negative and `1` is
positive. If the shared file does not exist, the data pipeline creates it once
at that centralized path. It never creates a project-local dataset folder.

The deterministic split is:

- 80% training
- 10% validation
- 10% test

The vocabulary is built from the training split only and saved as
`transformer_vocab.json`. Validation accuracy selects the best checkpoint. The
test split is evaluated only once after training.

## Tensor Shapes

- Input IDs: `[batch_size, sequence_length]`
- Embeddings: `[batch_size, sequence_length, embedding_dim]`
- Encoder output: `[batch_size, sequence_length, embedding_dim]`
- Mean-pooled output: `[batch_size, embedding_dim]`
- Classification logits: `[batch_size, 2]`

## Project Structure

```text
11_transformer_text_classification/
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

## Predict

After training:

```bash
python scripts/predict.py
```

This is a small educational Transformer Encoder classifier, not a pretrained
language model.

## Model Summary and Key Papers

### Historical Background

The Transformer replaced recurrence with self-attention and became the foundation of modern NLP. For classification, the encoder can model relationships between all tokens in parallel.

### Basic Structure

The model uses token embeddings, positional encodings, multi-head self-attention, feed-forward networks, residual connections, and layer normalization. A pooled representation is passed to a classifier.

### Why It Matters

Transformer encoders enabled scalable language representation learning and later powered models such as BERT and many modern text classifiers.

### Key Papers

* [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
