# LSTM Text Classification

This PyTorch project trains a Long Short-Term Memory network to classify short
text examples as negative or positive.

## LSTM

LSTM was introduced to address the difficulty vanilla RNNs have in learning
long-term dependencies. In addition to a hidden state, an LSTM maintains a
cell state that can carry information across many sequence steps.

The forget gate controls which cell-state information is removed, the input
gate controls which new information is stored, and the output gate controls
which information becomes the hidden state. These gates made LSTMs
historically important for language, speech, and time-series modeling.

LSTMs are more capable than vanilla RNNs for longer dependencies, although
attention-based models have largely replaced them in modern NLP systems.

## Dataset And Vocabulary

This project reuses the same small balanced sentiment dataset as the vanilla
RNN project:

```text
machine_learning/datasets/text_classification/simple_sentiment.csv
```

If the CSV is missing, the data module creates a small fallback dataset at the
same shared location. It does not create a project-local dataset copy.

Text is lowercased and split by whitespace. The vocabulary is built from the
training split only and includes `<pad>` and `<unk>` tokens. Sequences are
padded or truncated to the configured maximum length.

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

The classifier uses the final hidden state from the last LSTM layer. In
bidirectional mode, it concatenates the final forward and backward hidden
states, doubling the classifier input dimension.

## Project Structure

```text
08_lstm_text_classification/
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
`lstm_text_classification.pt`, then loads it and evaluates the test set once.

## Predict

After training:

```bash
python scripts/predict.py
```

The prediction script rebuilds the same training vocabulary, loads the best
checkpoint, and prints sentiment predictions for several example sentences.

## Model Summary and Key Papers

### Historical Background

LSTM was introduced to address the difficulty vanilla RNNs have with long-term dependencies. Its gates and cell state help preserve information over longer sequences.

### Basic Structure

An LSTM maintains both a hidden state and a cell state. Input, forget, and output gates control what information is stored, removed, and exposed to the next layer or classifier.

### Why It Matters

LSTM became a foundation for sequence modeling in language, speech, handwriting, and time series before attention-based architectures became dominant.

### Key Papers

* [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735)
