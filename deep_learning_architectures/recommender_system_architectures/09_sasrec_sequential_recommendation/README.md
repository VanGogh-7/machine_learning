# SASRec Sequential Recommendation

This project implements an educational PyTorch version of SASRec,
Self-Attentive Sequential Recommendation, on MovieLens 10M.

SASRec is historically important because it applies Transformer-style
self-attention to sequential recommendation. Earlier sequential recommenders
often used Markov chains, RNNs, or GRUs. SASRec uses causal self-attention to
model item-item dependencies in a user's ordered behavior sequence and predict
the next item.

Compared with earlier projects in this branch:

- NCF models user-item collaborative filtering.
- Two-Tower learns retrieval embeddings for candidate generation.
- DIN and DIEN use target-aware attention or interest evolution for CTR
  prediction.
- SASRec models sequential recommendation with causal self-attention.

This project is an educational implementation, not a full production recommender
system.

## Core Ideas

Sequential recommendation predicts what a user may interact with next based on
their ordered behavior history. A user behavior sequence is a timestamp-sorted
list of positive item interactions.

SASRec embeds item ids and positions, then passes the sequence through
Transformer blocks. Positional embeddings let the model distinguish early and
late sequence positions. Causal self-attention lets each position attend only to
current and previous items. A padding mask prevents padded item id `0` from
contributing to attention.

Each Transformer block contains causal multi-head self-attention, residual
connections, layer normalization, a feedforward network, and dropout. The final
non-padding hidden state becomes the sequence representation. A target item is
embedded separately, and the model scores the pair with a dot product.

This educational version uses binary next-item prediction with positive and
negative target items, so training uses `BCEWithLogitsLoss`. The model returns
raw logits because `BCEWithLogitsLoss` applies sigmoid internally in a
numerically stable way.

## Dataset

The raw dataset is MovieLens 10M:

```text
machine_learning/datasets/ml-10M100K/
```

Expected raw files:

```text
ratings.dat
movies.dat
```

The raw files use the MovieLens 10M `::` separator:

```text
userId::movieId::rating::timestamp
movieId::title::genres
```

Processed SASRec files are saved centrally under:

```text
machine_learning/datasets/ml-10M100K/processed_sasrec/
```

Expected processed files:

```text
train.csv
valid.csv
test.csv
feature_metadata.json
```

No dataset files are stored inside this project directory.

## Label Construction

The preprocessing step keeps positive interactions:

```text
rating >= positive_rating_threshold
```

For each user, positive interactions are sorted by timestamp. For each next item,
the previous items become the input sequence and the next item becomes a positive
target. Negative targets are sampled from items the user has not interacted with.

Each processed row contains:

```text
input_item_ids,target_item_id,label
```

`input_item_ids` is a left-padded, space-separated sequence with exactly
`max_sequence_length` values, for example:

```text
0 0 0 12 45 88 103
```

Item id `0` is reserved for padding. Encoded real item ids start from `1`.

## Data Preparation

Run preprocessing once:

```bash
python scripts/prepare_data.py
```

The training script does not process raw MovieLens 10M. If the processed files
are missing, it prints a clear message asking you to run the preparation step.

## Training

```bash
python scripts/train.py
```

The validation set is used for model selection. The best checkpoint is selected
by validation AUC when AUC is available, otherwise by validation loss. The test
set is evaluated only once after training finishes.

## Recommendation

```bash
python scripts/recommend.py
```

The recommendation script loads the saved checkpoint, reads a few sequences from
`test.csv`, scores candidate item ids in chunks for memory safety, and prints
top-k encoded item ids with logit scores. Movie titles are optional metadata and
are not required.

## Shapes

Inputs:

```text
input_item_ids: [batch_size, max_sequence_length]
target_item_ids: [batch_size]
```

Internal tensors:

```text
item embeddings: [batch_size, max_sequence_length, embedding_dim]
hidden states: [batch_size, max_sequence_length, embedding_dim]
sequence vector: [batch_size, embedding_dim]
target item embedding: [batch_size, embedding_dim]
```

Outputs:

```text
logits: [batch_size]
probabilities after sigmoid: [batch_size]
```

## Data Split

Samples are shuffled deterministically with `seed` and split into:

- training set
- validation set
- test set

With the default ratios, the split is 80% train, 10% validation, and 10% test.
The validation set is used for model selection. The test set is used only once
after training.

## Memory Safety

MovieLens 10M is large, so this project defaults to `debug_mode=True`.

Important config options in `src/config.py`:

- `debug_mode`
- `max_users`
- `max_interactions`
- `max_samples`
- `batch_size`
- `num_workers`
- `n_epochs`

Full preprocessing can be enabled by editing `src/config.py`:

```python
debug_mode = False
max_users = None
max_interactions = None
max_samples = None
```

Full MovieLens 10M preprocessing and training may take time and memory. The
negative sampler uses rejection sampling and does not build a dense user-item
matrix or any `num_users * num_items` candidate structure.

## Project Files

```text
scripts/prepare_data.py
scripts/train.py
scripts/recommend.py
src/config.py
src/data.py
src/engine.py
src/model.py
src/utils.py
src/visualize.py
```

## Model Summary and Key Papers

SASRec introduced a self-attention approach to sequential recommendation. Its
basic structure combines item embeddings, positional embeddings, causal
self-attention blocks, and next-item prediction. It matters because it brought
Transformer-style sequence modeling into recommendation systems and handled
longer-range item dependencies more flexibly than simple Markov or recurrent
methods.

Key paper:

- Self-Attentive Sequential Recommendation
  Authors: Wang-Cheng Kang, Julian McAuley
  Link: https://arxiv.org/abs/1808.09781
