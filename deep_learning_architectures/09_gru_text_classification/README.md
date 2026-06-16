# GRU Text Classification

This PyTorch project trains a Gated Recurrent Unit network to classify short
text examples as negative or positive.

## GRU

GRU is a gated recurrent architecture introduced as a simpler alternative to
LSTM. Its update gate controls how much previous hidden-state information is
kept, while its reset gate controls how much past information is used when
forming new candidate information.

Unlike LSTM, GRU does not maintain a separate cell state. It combines memory
and output information in a single hidden state, which usually gives GRU fewer
parameters and faster training than a comparable LSTM. GRU became historically
important as an effective recurrent model for language and sequence tasks,
although attention-based models later became dominant in many NLP systems.

## Dataset And Vocabulary

This project reuses the same small balanced sentiment dataset as the vanilla
RNN and LSTM projects:

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

The classifier uses the final hidden state from the last GRU layer. In
bidirectional mode, it concatenates the final forward and backward hidden
states, doubling the classifier input dimension.

## Project Structure

```text
09_gru_text_classification/
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
`gru_text_classification.pt`, then loads it and evaluates the test set once.

## Predict

After training:

```bash
python scripts/predict.py
```

The prediction script rebuilds the same training vocabulary, loads the best
checkpoint, and prints sentiment predictions for several example sentences.

## Model Summary and Key Papers

### Historical Background

GRU was introduced as a simpler gated recurrent architecture after LSTM. It keeps the gating idea but removes the separate cell state.

### Basic Structure

A GRU uses an update gate and reset gate to control information flow through the hidden state. For text classification, the final hidden state can summarize the sequence for a classifier.

### Why It Matters

GRU offered a practical alternative to LSTM that is often simpler, faster, and competitive on sequence modeling tasks.

### Key Papers

* [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
* [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555)
