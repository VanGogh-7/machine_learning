# Deep Interest Network CTR Prediction

This project implements an educational Deep Interest Network (DIN) model for
click-through-rate-style prediction using PyTorch.

## Deep Interest Network

DIN models a user's historical behavior sequence with target-aware attention:

- The **target item** is the candidate movie to score.
- The **user behavior sequence** is a fixed-length list of previous positive
  movie interactions.
- The **item embedding** layer maps target and historical item ids to vectors.
- The **activation unit**, also called the local activation unit, scores each
  historical item according to the target item.
- The **history mask** prevents padded history positions from contributing.
- The **user interest vector** is a weighted sum of historical item embeddings.
- The **prediction MLP** combines the target item embedding and user interest
  vector to produce one CTR-style logit.

DIN is historically important in recommender systems and advertising systems
because it introduced a target-aware mechanism for modeling diverse user
interests. A user's history should not be compressed into one fixed vector
independent of the target item. For an action movie target, action-related
history should usually receive higher attention than unrelated interactions.

Compared with Wide & Deep, DeepFM, DCN, and xDeepFM, DIN focuses less on general
feature crossing and more on behavior-sequence modeling. DIN is a natural next
step after those feature-interaction models and comes before DIEN in the
historical recommender-system architecture sequence.

The model returns raw logits instead of probabilities. Training uses
`BCEWithLogitsLoss`, which combines sigmoid and binary cross-entropy in a
numerically stable operation. Probabilities are computed with sigmoid only for
metrics and prediction output.

## Dataset

This project uses MovieLens as an educational sequential recommendation dataset
and transforms it into a CTR-style binary prediction task.

Raw MovieLens 10M files are expected under the centralized dataset directory:

```text
machine_learning/datasets/ml-10M100K/
```

Required raw files:

```text
ratings.dat
movies.dat
```

MovieLens 10M uses `.dat` files with `::` separators:

```text
userId::movieId::rating::timestamp
movieId::title::genres
```

The raw files are converted once into DIN-ready CSV files under:

```text
machine_learning/datasets/ml-10M100K/processed_din/
```

Training reads these processed files directly. It does not rebuild the raw
MovieLens sequence dataset each time `scripts/train.py` runs.

## CTR-Style Sample Construction

For each user:

1. Interactions are sorted by timestamp.
2. Ratings greater than or equal to `positive_rating_threshold` are treated as
   positive interactions.
3. Previous positive items become the historical behavior sequence.
4. The next positive item becomes a positive target item.
5. Negative target items are sampled from movies the user has not interacted
   with.

Each sample contains:

- `target_item_id`
- `history_item_ids`
- `history_mask`
- `label`

Labels are binary: `1` means positive or clicked/interacted, and `0` means a
sampled negative item. This is an educational approximation of DIN's original
industrial display advertising setting.

Histories are left-padded with item id `0`, so the most recent items stay near
the end of each sequence. Samples are deterministically shuffled and split into
training, validation, and test sets. Validation data is used for model
selection. Test data is evaluated only once after training finishes.

## Tensor Shapes

- Target item ids: `[batch_size]`
- History item ids: `[batch_size, max_history_length]`
- History mask: `[batch_size, max_history_length]`
- Target item embedding: `[batch_size, embedding_dim]`
- History item embeddings: `[batch_size, max_history_length, embedding_dim]`
- Attention weights: `[batch_size, max_history_length]`
- User interest vector: `[batch_size, embedding_dim]`
- Raw logits: `[batch_size]`
- Probabilities after sigmoid: `[batch_size]`

## Project Structure

```text
05_din_ctr_prediction/
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

The saved files include the best model checkpoint, preprocessing metadata, and
training history.

## Data Preparation

Step 1 converts raw MovieLens 10M files into compact DIN-ready CSV files:

```bash
python scripts/prepare_data.py
```

This creates:

```text
datasets/ml-10M100K/processed_din/train.csv
datasets/ml-10M100K/processed_din/valid.csv
datasets/ml-10M100K/processed_din/test.csv
datasets/ml-10M100K/processed_din/feature_metadata.json
```

The processed CSV schema is:

```text
target_item_id,history_item_ids,history_mask,label
```

`history_item_ids` and `history_mask` are space-separated fixed-length
sequences, for example:

```text
305,"0 0 0 12 45","0 0 0 1 1",1
```

## Train

Step 2 trains from the processed CSV files:

```bash
python scripts/train.py
```

If the processed files are missing, training stops and asks you to run
`python scripts/prepare_data.py`.

The best checkpoint is selected by validation AUC when AUC can be computed. If a
split contains only one class and AUC is unavailable, validation loss is used.
The test split is not used for checkpoint selection.

## Predict

Step 3 runs a few predictions from the processed test CSV:

```bash
python scripts/predict.py
```

If movie-title metadata was not saved, prediction output falls back to encoded
item ids.

## Memory Safety

MovieLens 10M is much larger than `ml-latest-small`. The project defaults to
`debug_mode=True` in `src/config.py`, so the first preprocessing run is limited
and safe for a normal laptop.

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

Full preprocessing can take substantial time and memory. The preprocessing code
uses rejection sampling for negative examples and does not build a full
user-to-negative-candidates dictionary.

This project is an educational PyTorch implementation, not a full production
CTR system.

## Model Summary and Key Papers

### Historical Background

DIN was introduced for click-through rate prediction in industrial
recommendation and advertising systems. It addressed the problem that users have
diverse interests, and different parts of a user's behavior history should be
activated for different target ads or items.

### Basic Structure

DIN embeds the target item and historical behavior items, uses a local
activation unit to compute target-aware attention over the history, forms a
user interest vector, and feeds the interest vector plus target embedding into a
prediction MLP.

### Why It Matters

DIN shifted the focus from fixed user representations to target-aware behavior
sequence modeling. This made it a key step toward later sequential and
interest-evolution models such as DIEN.

### Key Papers

- [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)
  by Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Ying Fan, Han Zhu, Xiao Ma,
  Yanghui Yan, Junqi Jin, Han Li, and Kun Gai.
