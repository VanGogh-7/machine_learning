# Model Roadmap

This roadmap explains how the ten projects relate historically and
conceptually. The order is chosen as a practical learning route rather than a
strict publication timeline.

## 1. From Manual Feature Engineering to Deep Feature Interaction

Wide & Deep starts with a clear split between memorization and generalization.
The wide side can memorize sparse feature crosses, while the deep side learns
from dense embeddings of sparse categorical features.

DeepFM reduces the need to manually design wide crosses. It combines an
FM-style component for first-order and second-order feature interactions with a
deep neural network for higher-order implicit interactions.

DCN introduces the Cross Network, which explicitly learns bounded-degree feature
crosses. It is useful because it gives feature interaction learning a structured
form instead of relying only on a generic MLP.

xDeepFM adds the Compressed Interaction Network, CIN, to model vector-wise
explicit feature interactions while still keeping a deep component for implicit
interactions.

Key ideas:

- memorization and generalization
- sparse categorical features
- embeddings
- explicit and implicit feature crossing
- FM-style interaction
- Cross Network
- CIN

## 2. From Feature Interaction to User Interest Modeling

DIN shifts attention from static feature interaction to user behavior history.
It uses target-aware attention so the same user history can produce different
interest representations for different target items.

DIEN extends this idea by modeling interest evolution. It uses GRU-style
sequence modeling, an auxiliary loss to supervise interest states, and AUGRU to
make interest evolution target-aware.

Key ideas:

- user behavior history
- target-aware attention
- activation unit
- interest evolution
- GRU
- auxiliary loss
- AUGRU

## 3. From Interaction Prediction to Collaborative Filtering

NCF focuses on collaborative filtering from user-item interaction data. Classical
matrix factorization uses a fixed dot product between user embeddings and item
embeddings. NCF generalizes this with neural interaction functions.

The educational NeuMF implementation combines GMF, which keeps the elementwise
matrix-factorization signal, with an MLP interaction branch for nonlinear
patterns.

Key ideas:

- matrix factorization
- user embeddings
- item embeddings
- GMF
- MLP interaction
- NeuMF

## 4. From Ranking to Large-Scale Retrieval

Two-Tower Retrieval separates the model into a user tower and an item tower. The
two towers produce retrieval embeddings that can be compared with a dot product.

This structure is important because item embeddings can be computed offline and
searched efficiently during candidate generation. In-batch negatives provide a
simple educational training signal without permanently storing negative samples.

Key ideas:

- user tower
- item tower
- candidate generation
- retrieval embedding
- in-batch negatives
- offline item embedding index

## 5. From RNN-Style Sequences to Self-Attentive Sequential Recommendation

SASRec applies Transformer-style causal self-attention to sequential
recommendation. Instead of summarizing user behavior only with recurrent state,
it lets each sequence position attend to previous items.

The model uses positional embeddings, causal masks, padding masks, and a final
sequence representation to score candidate next items.

Key ideas:

- next-item prediction
- causal self-attention
- positional embeddings
- sequence representation

## 6. From User-Item Pairs to User-Item Graphs

LightGCN treats recommendation as propagation over a user-item bipartite graph.
Users and items are graph nodes, and positive interactions are graph edges.

LightGCN removes the feature transformations and nonlinear activations common in
earlier GCN recommenders. It keeps the essential operation: normalized
neighborhood aggregation. Training uses BPR loss, and evaluation uses ranking
metrics such as Recall@K and NDCG@K.

Key ideas:

- user-item bipartite graph
- graph propagation
- normalized adjacency
- BPR loss
- Recall@K
- NDCG@K

## 7. What This Collection Does Not Yet Cover

Useful future extensions could include:

- PNN
- AFM
- AutoInt
- BST
- BERT4Rec
- YouTube DNN
- DLRM
- DCNv2
- MMoE
- PLE

These are not created in this cleanup pass.
