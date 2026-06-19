# Two-Tower Retrieval for Candidate Generation

This project implements an educational PyTorch version of a Two-Tower Retrieval
model for candidate generation on MovieLens 10M.

Two-Tower models are historically and practically important because they split a
recommender into two independently computable embedding functions:

- a user tower that maps user ids or user features into a user embedding
- an item tower that maps item ids or item features into an item embedding
- a similarity function, usually dot product or cosine similarity

This is useful for large-scale recommender systems because item embeddings can be
precomputed offline. At serving time, the system only needs to encode the user
and search for nearby item embeddings, instead of scoring every user-item pair
through a large joint model.

## Retrieval Concepts

Candidate generation is the first broad retrieval stage of a recommender system.
It finds a small set of likely relevant items from a large corpus. A later
ranking stage can then use a heavier model to sort those candidates with richer
features.

The user tower produces retrieval embeddings for users. The item tower produces
retrieval embeddings for items. Dot product similarity compares the two vectors.
Higher scores mean the item is closer to the user in the learned retrieval space.

This project trains with in-batch negatives. For a batch of positive user-item
pairs, every other item in the batch acts as a negative item for each user. The
model builds a similarity matrix:

```text
scores = user_vectors @ item_vectors.T
```

The shape is:

```text
[batch_size, batch_size]
```

`scores[i, j]` is the similarity between user `i` and item `j` in the current
batch. The diagonal entries are the positive user-item pairs. Training uses
`CrossEntropyLoss` with labels:

```python
torch.arange(batch_size)
```

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

Processed Two-Tower files are saved centrally under:

```text
machine_learning/datasets/ml-10M100K/processed_two_tower/
```

Expected processed files:

```text
train.csv
valid.csv
test.csv
feature_metadata.json
```

No dataset files are stored inside this project directory.

## Positive Pair Construction

The preprocessing step keeps positive interactions only:

```text
rating >= positive_rating_threshold
```

Each processed row contains:

```text
user_id,item_id
```

Negatives are not written to disk. They are handled by in-batch negatives during
training.

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
by validation Recall@10, with validation loss used as the tie-breaker. The test
set is evaluated only once after training finishes.

## Build Item Index

```bash
python scripts/build_item_index.py
```

This script loads the best checkpoint, encodes all item ids from `0` to
`num_items - 1`, and saves:

```text
outputs/item_ids.npy
outputs/item_embeddings.npy
```

This educational project does not use FAISS or external approximate nearest
neighbor libraries.

## Retrieval

```bash
python scripts/retrieve.py
```

The retrieval script loads the checkpoint, loads the saved item embeddings,
encodes a few users from the test set, computes dot product scores against all
saved item embeddings, and prints top-k encoded item ids with similarity scores.
Movie titles are optional metadata and are not required.

## Shapes

Inputs:

```text
user_ids: [batch_size]
item_ids: [batch_size]
```

Internal tensors:

```text
user_vectors: [batch_size, output_dim]
item_vectors: [batch_size, output_dim]
scores: [batch_size, batch_size]
```

`scores[i, j]` represents the similarity between user `i` and item `j` in the
current batch. The diagonal entries are the positive pairs.

## Data Split

Positive pairs are shuffled deterministically with `seed` and split into:

- training set
- validation set
- test set

With the default ratios, the split is 80% train, 10% validation, and 10% test.

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

Full MovieLens 10M preprocessing and training may take time and memory. This
project stores only positive pairs and uses in-batch negatives, so it does not
build a dense user-item matrix or a large negative-sample table.

## Difference From NCF

NCF learns a joint user-item interaction model. A joint model is useful for
learning flexible interaction functions, but scoring a large item corpus requires
running the joint model many times.

Two-Tower learns separate user and item embeddings. Item vectors can be
precomputed and searched efficiently, which makes this architecture better suited
to candidate retrieval.

This project is an educational PyTorch implementation, not a full production
retrieval system.

## Project Files

```text
scripts/prepare_data.py
scripts/train.py
scripts/build_item_index.py
scripts/retrieve.py
src/config.py
src/data.py
src/engine.py
src/model.py
src/utils.py
src/visualize.py
```

## Model Summary and Key Papers

Two-Tower retrieval models grew out of the need to retrieve candidates from very
large item corpora. Their basic structure learns one encoder for users and one
encoder for items, then compares the resulting retrieval embeddings with a fast
similarity operation. They matter because they make offline item embedding
precomputation and online nearest-neighbor retrieval practical.

Key references:

- Deep Neural Networks for YouTube Recommendations
  Authors: Paul Covington, Jay Adams, Emre Sargin
  Link: https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/
- Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations
  Authors: Yi Yang, Wen-tau Yih, Christopher Meek
  Link: https://dl.acm.org/doi/10.1145/2783258.2783406
