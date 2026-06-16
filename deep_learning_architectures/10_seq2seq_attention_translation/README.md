# Seq2Seq Attention Translation

This PyTorch project implements an English-to-German sequence-to-sequence
translation model with Bahdanau additive attention.

## Seq2Seq And Attention

Seq2Seq encoder-decoder models map an input sequence to an output sequence.
Early models compressed the entire source sentence into one fixed-length
vector, which made long sentences difficult to translate.

Attention allows the decoder to dynamically focus on different encoder hidden
states at each output step. Bahdanau additive attention scores each source
position by combining the current decoder hidden state with each encoder
output through learned linear projections and a nonlinear activation.

Attention-based Seq2Seq models were historically important for neural machine
translation and were later generalized by the Transformer architecture.

## Dataset

The project uses the real Multi30k English-German parallel dataset through a
manual text-file loader. Dataset files are shared across the repository under:

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

The loader does not silently create a toy dataset. If the files are missing,
it reports the expected paths clearly.

The training files are used for training and vocabulary construction. The
validation files are used for model selection based on validation loss. The
test files are evaluated only once after training.

## Data And Tensor Shapes

English is the source language and German is the target language. Separate
source and target vocabularies are built from the training files only.

- Encoder input: `[batch_size, src_seq_len]`
- Decoder input: `[batch_size, tgt_seq_len - 1]`
- Output logits: `[batch_size, tgt_seq_len - 1, target_vocab_size]`

For a target sequence `<bos> ich mag hunde <eos>`, teacher forcing feeds
`<bos> ich mag hunde` into the decoder and trains it to predict
`ich mag hunde <eos>`. During training, teacher forcing sometimes supplies the
real previous target token instead of the decoder's prediction.

## Project Structure

```text
10_seq2seq_attention_translation/
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

Training saves the checkpoint with the lowest validation loss as
`seq2seq_attention_translation.pt`, then loads it and evaluates the test set
once.

## Predict

After training:

```bash
python scripts/predict.py
```

Prediction rebuilds both vocabularies from the training files and greedily
translates several English examples from the test set.
