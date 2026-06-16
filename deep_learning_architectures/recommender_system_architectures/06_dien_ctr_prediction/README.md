# Deep Interest Evolution Network CTR Prediction

This project implements an educational Deep Interest Evolution Network, DIEN,
for CTR-style sequential recommendation using PyTorch.

DIEN is a natural next step after DIN. DIN uses target-aware attention over a
user's historical behavior sequence, but it does not explicitly model how user
interests evolve over time. DIEN adds an interest extraction GRU, an auxiliary
next-behavior loss, and an attention-updated GRU, AUGRU, to model interest
evolution with respect to the target item.

## DIEN Concepts

- **CTR prediction** estimates whether a user will click or interact with a
  target item.
- The **user behavior sequence** is a fixed-length list of previous positive
  movie interactions.
- The **target item** is the candidate movie being scored.
- **Item embeddings** map target and historical item ids to dense vectors.
- **Interest extraction** uses a GRU over historical behavior embeddings to
  create latent interest states.
- The **auxiliary loss** asks each extracted interest state to predict the next
  observed behavior. This project uses a simple positive next-item objective as
  an educational approximation of DIEN's original auxiliary loss.
- **Target-aware attention** scores each interest state against the target item.
- **AUGRU** scales the GRU update gate by target-aware attention weights.
- **Interest evolution** produces a target-conditioned user interest vector.
- The **prediction MLP** combines the evolved interest vector with the target
  item embedding and outputs one raw CTR logit.

The model returns raw logits instead of probabilities. Training uses
`BCEWithLogitsLoss`, which combines sigmoid and binary cross-entropy in a
numerically stable operation. Sigmoid is used only for metrics and prediction
output.

## Dataset

This project uses MovieLens 10M as an educational sequential recommendation
dataset and transforms it into a CTR-style binary prediction task. This is an
educational approximation of DIEN's original industrial display advertising
setting.

Expected raw dataset location:

```text
machine_learning/datasets/ml-10M100K/
```

Expected raw files:

```text
ratings.dat
movies.dat
```

MovieLens 10M uses `.dat` files with `::` separators:

```text
userId::movieId::rating::timestamp
movieId::title::genres
```

The raw files are converted once into DIEN-ready CSV files under:

```text
machine_learning/datasets/ml-10M100K/processed_dien/
```

Expected processed files:

```text
train.csv
valid.csv
test.csv
feature_metadata.json
```

Training reads these processed files directly. It does not process raw
MovieLens 10M inside `scripts/train.py`.

## Sample Construction

For each user:

1. Interactions are sorted by timestamp.
2. Ratings greater than or equal to `positive_rating_threshold` are positive.
3. Previous positive items become the historical behavior sequence.
4. The next positive item becomes a positive target.
5. Negative targets are sampled from items the user has not interacted with.
6. `next_history_item_ids` stores shifted next-item supervision for DIEN's
   auxiliary loss.

Each processed CSV row contains:

```text
target_item_id,history_item_ids,next_history_item_ids,history_mask,aux_mask,label
```

Sequence columns are fixed-length, space-separated integers. Histories are
left-padded with item id `0`, so the most recent behaviors stay near the end of
the sequence.

Labels are binary:

- `1`: positive high-rating interaction
- `0`: sampled negative item

## Data Preparation

Step 1 converts raw MovieLens 10M files into compact DIEN-ready CSV files:

```bash
python scripts/prepare_data.py
```

## Train

Step 2 trains from the processed CSV files:

```bash
python scripts/train.py
```

The validation set is used for checkpoint selection. The test set is evaluated
only once after training finishes and is not used to save the best checkpoint.

## Predict

Step 3 runs a few predictions from the processed test CSV:

```bash
python scripts/predict.py
```

If movie-title metadata was not saved, prediction output falls back to encoded
item ids.

## Tensor Shapes

- Target item ids: `[batch_size]`
- History item ids: `[batch_size, max_history_length]`
- Next history item ids: `[batch_size, max_history_length]`
- History mask: `[batch_size, max_history_length]`
- Auxiliary mask: `[batch_size, max_history_length]`
- Target item embedding: `[batch_size, embedding_dim]`
- History item embeddings: `[batch_size, max_history_length, embedding_dim]`
- Interest states: `[batch_size, max_history_length, gru_hidden_dim]`
- Attention weights: `[batch_size, max_history_length]`
- Evolved interest: `[batch_size, gru_hidden_dim]`
- Raw logits: `[batch_size]`
- Probabilities after sigmoid: `[batch_size]`

## Memory Safety

MovieLens 10M is large. The project defaults to `debug_mode=True` in
`src/config.py`, so the first preprocessing and training run is limited and safe
for a normal laptop.

Important defaults:

```python
debug_mode = True
max_users = 2000
max_interactions = 300000
max_samples = 100000
batch_size = 512
num_workers = 0
n_epochs = 3
```

For full MovieLens 10M preprocessing, edit `src/config.py` manually:

```python
debug_mode = False
max_users = None
max_interactions = None
max_samples = None
```

Full preprocessing and training can take substantial time and memory. The
preprocessing code uses rejection sampling for negative examples and does not
build a full user-to-negative-candidates dictionary.

This project is an educational PyTorch implementation, not a full production
CTR system.

## Project Structure

```text
06_dien_ctr_prediction/
├── scripts/
│   ├── prepare_data.py
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

Training outputs are saved under:

```text
outputs/
```

## Model Summary and Key Papers

### Historical Background

DIEN was proposed for CTR prediction in industrial recommendation and
advertising systems. It was designed to capture the evolution of user interests
instead of treating historical behaviors as an unordered or static sequence.

### Basic Structure

DIEN embeds target and historical items, extracts interest states with a GRU,
uses an auxiliary next-behavior loss to improve interest extraction, applies
target-aware attention, evolves interests with AUGRU, and predicts CTR with an
MLP.

### Why It Matters

DIEN made user interest evolution explicit. It connected sequential modeling
with target-conditioned recommendation and influenced later behavior-sequence
recommendation models.

### Key Paper

- [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/abs/1809.03672)
  by Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang
  Zhu, and Kun Gai.
