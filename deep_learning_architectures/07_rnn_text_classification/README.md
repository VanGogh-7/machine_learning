# Vanilla RNN Text Classification

This PyTorch project trains a vanilla recurrent neural network to classify
short text examples as negative or positive.

## Recurrent Neural Networks

Recurrent Neural Networks process sequences one element at a time while
carrying a hidden state forward. They were among the earliest neural network
families designed specifically for sequential data and became important for
language, speech, and time-series tasks.

This project converts token ids into embeddings, processes the embedding
sequence with `nn.RNN`, and uses the final hidden state to classify the whole
sentence. Vanilla RNNs can struggle with long-term dependencies, which
motivates architectures such as LSTM and GRU.

## Dataset And Vocabulary

The project uses a small balanced sentiment CSV with two labels:

- `0`: negative
- `1`: positive

The shared dataset is stored under:

```text
machine_learning/datasets/text_classification/simple_sentiment.csv
```

If the CSV is missing, the data module creates a small fallback dataset
automatically. Text is lowercased and split by whitespace. The vocabulary is
built from the training split only and includes `<pad>` and `<unk>` tokens.
Sequences are padded or truncated to the configured maximum length.

The deterministic data split is:

- 80% training
- 10% validation
- 10% test

The validation set is used for model selection. The test set is evaluated only
once after training.

## Tensor Shapes

- Input token ids: `[batch_size, sequence_length]`
- Embeddings: `[batch_size, sequence_length, embedding_dim]`
- Output logits: `[batch_size, 2]`

## Project Structure

```text
07_rnn_text_classification/
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

## Setup

From the project root:

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py
```

Training saves the checkpoint with the best validation accuracy as
`rnn_text_classification.pt`, then loads it and evaluates the test set once.

## Predict

After training:

```bash
python scripts/predict.py
```

The prediction script rebuilds the same training vocabulary, loads the best
checkpoint, and prints sentiment predictions for several example sentences.

## Model Summary and Key Papers

### Historical Background

Vanilla RNNs are classic neural networks for sequential data. They were important before gated recurrent models and attention-based models became dominant.

### Basic Structure

An RNN processes tokens one step at a time while maintaining a hidden state. For text classification, the final hidden state or a pooled sequence representation can be used as the input to a classifier.

### Why It Matters

RNNs introduced a simple neural way to model ordered data and temporal dependencies, even though they struggle with long-range dependencies.

### Key Papers

* [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0)
* [Finding Structure in Time](https://doi.org/10.1207/s15516709cog1402_1)
