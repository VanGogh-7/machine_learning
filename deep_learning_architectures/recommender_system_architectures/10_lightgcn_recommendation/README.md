# LightGCN Recommendation

This project implements an educational PyTorch version of LightGCN, Light Graph
Convolution Network, for implicit-feedback recommendation on MovieLens 10M.

LightGCN is historically important because it simplifies earlier graph neural
network recommenders. Earlier GCN-based recommenders often used feature
transformation matrices, nonlinear activations, and complex message passing.
LightGCN argues that for collaborative filtering, the essential component is
neighborhood aggregation over the user-item interaction graph.

This project is an educational implementation, not a full production graph
recommender system.

## Core Ideas

Collaborative filtering learns recommendations from user-item interactions.
Implicit feedback treats observed interactions as positive preference signals,
even when explicit ratings are not used directly by the model.

LightGCN builds a user-item bipartite graph. User nodes connect to item nodes
when a positive interaction exists. The graph uses:

```text
user node: user_id
item node: num_users + item_id
```

The normalized adjacency matrix is:

```text
A_hat = D^{-1/2} A D^{-1/2}
```

LightGCN propagates embeddings with sparse graph multiplication:

```text
all_embeddings = A_hat @ all_embeddings
```

There are no feature transformation matrices and no nonlinear activations inside
the propagation layers. Embeddings from layer 0 through layer K are averaged to
produce the final user and item embeddings.

Training uses BPR loss. BPR compares a positive item against a sampled negative
item for the same user and encourages the positive score to be higher.

Evaluation reports Recall@K and NDCG@K. Recall@K measures whether relevant items
appear in the top K. NDCG@K also rewards ranking relevant items near the top.

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

Processed LightGCN files are saved centrally under:

```text
machine_learning/datasets/ml-10M100K/processed_lightgcn/
```

Expected processed files:

```text
train.csv
valid.csv
test.csv
feature_metadata.json
```

No dataset files are stored inside this project directory.

## Edge Construction

The preprocessing step keeps positive interactions:

```text
rating >= positive_rating_threshold
```

Each processed row contains:

```text
user_id,item_id
```

The graph is built only from `train.csv`. Validation and test edges are never
used in graph construction.

Negative items are not stored in CSV files. They are sampled dynamically during
BPR training with rejection sampling from items the user has not interacted with
in the training set.

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
by validation Recall@K. The test set is evaluated only once after training.

## Recommendation

```bash
python scripts/recommend.py
```

The recommendation script loads the checkpoint, rebuilds the graph from training
interactions only, scores all items for a few users from `test.csv`, excludes
items seen in training, and prints top-k encoded item ids with scores. Movie
titles are optional metadata and are not required.

## Shapes

Training inputs:

```text
user_ids: [batch_size]
positive_item_ids: [batch_size]
negative_item_ids: [batch_size]
```

Internal tensors:

```text
all embeddings: [num_users + num_items, embedding_dim]
final user embeddings: [num_users, embedding_dim]
final item embeddings: [num_items, embedding_dim]
positive scores: [batch_size]
negative scores: [batch_size]
```

## Data Split

Positive edges are shuffled deterministically with `seed` and split into:

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
- `max_edges`
- `batch_size`
- `num_workers`
- `n_epochs`
- `eval_max_users`

Full preprocessing can be enabled by editing `src/config.py`:

```python
debug_mode = False
max_users = None
max_interactions = None
max_edges = None
```

Full MovieLens 10M graph construction and evaluation may take time and memory.
This project uses native PyTorch sparse tensors for graph propagation and avoids
dense user-item matrices and full negative-candidate lists.

## Difference From Earlier Projects

- NCF learns neural user-item interaction functions.
- Two-Tower learns separate retrieval embeddings.
- SASRec models sequential behavior with self-attention.
- LightGCN directly propagates embeddings over the user-item interaction graph.

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

LightGCN simplified graph collaborative filtering by removing transformations
and nonlinear activations from GCN propagation. Its basic structure learns user
and item embeddings, propagates them over the normalized bipartite interaction
graph, averages layer embeddings, and trains with BPR ranking loss. It matters
because it showed that simpler graph propagation can be highly effective for
collaborative filtering.

Key paper:

- LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
  Authors: Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang
  Link: https://arxiv.org/abs/2002.02126
