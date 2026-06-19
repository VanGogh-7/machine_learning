# Recommender System Architecture Collection

This folder contains educational PyTorch implementations of historically
important deep learning models in recommender systems. The projects are designed
as a learning path, with centralized datasets, clean project boundaries, and
debug-friendly defaults for large MovieLens 10M experiments.

The sequence is not the only possible historical order, but it is a useful route
from CTR feature interaction models to interest modeling, collaborative
filtering, retrieval, sequential recommendation, and graph recommendation:

```text
Wide & Deep -> DeepFM -> DCN -> xDeepFM -> DIN -> DIEN -> NCF -> Two-Tower -> SASRec -> LightGCN
```

## Model Table

| Index | Project | Model | Main idea | Recommendation task | Dataset | Key paper |
|---|---|---|---|---|---|---|
| 01 | `01_wide_deep_ctr_prediction` | Wide & Deep | Combine memorization from wide features with generalization from deep embeddings. | CTR prediction | Criteo-style CTR | Wide & Deep Learning for Recommender Systems |
| 02 | `02_deepfm_ctr_prediction` | DeepFM | Combine FM-style feature interactions with a deep network. | CTR prediction | Criteo-style CTR | DeepFM |
| 03 | `03_dcn_ctr_prediction` | DCN | Learn explicit bounded-degree feature crosses with a Cross Network. | CTR prediction | Criteo-style CTR | Deep & Cross Network |
| 04 | `04_xdeepfm_ctr_prediction` | xDeepFM | Learn explicit vector-wise crosses with CIN plus implicit deep interactions. | CTR prediction | Criteo-style CTR | xDeepFM |
| 05 | `05_din_ctr_prediction` | DIN | Use target-aware attention over user behavior history. | CTR prediction with behavior history | MovieLens 10M | Deep Interest Network |
| 06 | `06_dien_ctr_prediction` | DIEN | Model interest evolution with GRU/AUGRU and auxiliary supervision. | CTR prediction with interest evolution | MovieLens 10M | Deep Interest Evolution Network |
| 07 | `07_ncf_collaborative_filtering` | NCF / NeuMF | Replace fixed matrix-factorization dot products with neural interaction functions. | Implicit collaborative filtering | MovieLens 10M | Neural Collaborative Filtering |
| 08 | `08_two_tower_retrieval` | Two-Tower Retrieval | Learn separate user and item embeddings for candidate retrieval. | Candidate generation / retrieval | MovieLens 10M | YouTube DNN; large-corpus neural retrieval |
| 09 | `09_sasrec_sequential_recommendation` | SASRec | Use causal self-attention for next-item sequential recommendation. | Sequential next-item prediction | MovieLens 10M | Self-Attentive Sequential Recommendation |
| 10 | `10_lightgcn_recommendation` | LightGCN | Propagate embeddings over the user-item graph without nonlinear GCN layers. | Graph collaborative filtering | MovieLens 10M | LightGCN |

## Major Groups

### 1. CTR Feature Interaction Models

`01_wide_deep_ctr_prediction`, `02_deepfm_ctr_prediction`,
`03_dcn_ctr_prediction`, and `04_xdeepfm_ctr_prediction` focus on click-through
rate prediction from sparse categorical and numerical features. They show how
deep recommender systems moved from manually engineered crosses toward learned
feature interactions.

### 2. User Interest and Behavior Sequence Models

`05_din_ctr_prediction`, `06_dien_ctr_prediction`, and
`09_sasrec_sequential_recommendation` focus on user behavior histories. DIN uses
target-aware attention, DIEN models evolving interests with recurrent units, and
SASRec uses causal self-attention for next-item prediction.

### 3. Collaborative Filtering and Retrieval Models

`07_ncf_collaborative_filtering`, `08_two_tower_retrieval`, and
`10_lightgcn_recommendation` focus on interaction data. NCF learns neural
user-item interaction functions, Two-Tower learns retrieval embeddings, and
LightGCN propagates embeddings over the user-item graph.

## Dataset Rule

All datasets stay under:

```text
machine_learning/datasets/
```

Subprojects should not create local dataset folders. MovieLens-based projects
use `scripts/prepare_data.py` to create compact processed files once, then
`scripts/train.py` reads those processed files directly.

## Debug Safety

MovieLens 10M is large. MovieLens-based projects default to `debug_mode=True`
and safety limits such as `max_users`, `max_interactions`, `max_samples`,
`max_edges`, and `eval_max_users`. Do not disable debug mode until the debug run
works on your machine.

See also:

- `MODEL_ROADMAP.md` for the conceptual learning path.
- `DATASETS.md` for dataset locations and processed-file rules.
- `RUN_ALL_DEBUG.md` for minimal debug commands.
- `PROJECT_STATUS.md` for a file-level project checklist.
