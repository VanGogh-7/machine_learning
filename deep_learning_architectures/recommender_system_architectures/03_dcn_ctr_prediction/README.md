# Deep & Cross Network CTR Prediction

This project implements an educational Deep & Cross Network (DCN) model for
click-through-rate (CTR) prediction using PyTorch.

## Deep & Cross Network

DCN combines explicit feature crossing with a standard deep neural network:

- The **embedding layer** converts sparse categorical feature IDs into dense
  vectors and concatenates them with numerical features.
- The original input vector `x0` contains numerical features and all categorical
  embeddings.
- The **Cross Network** applies explicit feature crossing with the original DCN
  cross layer formula.
- The **Deep Network** learns nonlinear interactions implicitly through an MLP.
- The final prediction concatenates the cross output and deep output, then maps
  the combined representation to one scalar logit.

DCN is historically important in recommender systems, advertising systems, and
CTR prediction because it reduces the need for manually designed cross features.
Compared with Wide & Deep, the cross network learns structured feature crosses
directly. Compared with DeepFM, DCN uses explicit cross layers instead of the
factorization-machine second-order interaction formula.

CTR prediction estimates whether a user will click an item or advertisement.
The label is binary: `0` means not clicked and `1` means clicked. Numerical
features usually represent continuous or count-based signals. Sparse categorical
features represent high-cardinality fields such as user, item, ad, context, or
category IDs.

The cross layer formula is:

```text
x_{l+1} = x_0 * (x_l w_l) + b_l + x_l
```

Here `x_0` is the original input, `x_l` is the current cross layer input,
`w_l` is a learnable vector, `b_l` is a learnable bias vector, and `*` is
elementwise multiplication with broadcasting. This structure models bounded-
degree feature interactions in a parameter-efficient way.

The model returns raw logits instead of probabilities. Training uses
`BCEWithLogitsLoss`, which combines sigmoid and binary cross-entropy in a
numerically stable operation. Probabilities are computed with sigmoid only for
metrics and prediction output.

DCN is a natural next step after Wide & Deep and DeepFM, and before xDeepFM,
DIN, DIEN, and other recommender-system architectures.

## Dataset

Use the same real Criteo-style CTR dataset as the previous recommender-system
projects, stored centrally at:

```text
machine_learning/datasets/criteo_ctr/
```

Expected processed files:

```text
train.csv
valid.csv
test.csv
```

Common Criteo-style columns are:

```text
label
I1, I2, ..., I13
C1, C2, ..., C26
```

If all three processed files exist, the project uses them directly. If exactly
one processed CSV exists in the dataset directory, it is deterministically split
into 80% training, 10% validation, and 10% test subsets using `config.seed`.

If a common raw Criteo TSV such as `train.txt`, `train.tsv`, or
`dac_sample.txt` exists under `machine_learning/datasets/criteo_ctr/`, the data
pipeline reads it with the standard Criteo column layout and writes centralized
processed `train.csv`, `valid.csv`, and `test.csv` files.

If no usable data exists, the project prints a clear error asking you to place
the processed files under `machine_learning/datasets/criteo_ctr/`. It does not
create a toy dataset and it does not create local dataset folders inside this
project.

All numerical normalization statistics and categorical vocabularies are fitted
from the training split only. Validation data is used for model selection. Test
data is evaluated only once after training finishes.

## Tensor Shapes

- Numerical features: `[batch_size, num_numerical_features]`
- Categorical features: `[batch_size, num_categorical_features]`
- Original input `x0`: `[batch_size, input_dim]`
- Cross Network output: `[batch_size, input_dim]`
- Deep Network output: `[batch_size, deep_last_dim]`
- Raw logits: `[batch_size]`
- Probabilities after sigmoid: `[batch_size]`

## Project Structure

```text
03_dcn_ctr_prediction/
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

Training outputs are saved under:

```text
outputs/
```

The saved files include the best model checkpoint, preprocessing metadata, and
training history.

## Train

```bash
python scripts/train.py
```

The best checkpoint is selected by validation AUC when AUC can be computed. If a
split contains only one class and AUC is unavailable, validation loss is used.
The test split is not used for checkpoint selection.

## Predict

After training:

```bash
python scripts/predict.py
```

The prediction script loads the saved checkpoint and feature metadata, applies
the same numerical normalization and categorical ID mapping used in training,
and prints predicted click probabilities for a few rows.

This project is an educational PyTorch implementation, not a full production
CTR system.

## Model Summary and Key Papers

### Historical Background

Deep & Cross Network was introduced for ad click prediction and recommender
ranking tasks where feature crosses matter but manual cross-feature engineering
is expensive and brittle.

### Basic Structure

DCN starts with an embedding and stacking layer that creates `x0`. A Cross
Network explicitly applies feature crossing at each layer, while a Deep Network
learns nonlinear interactions. The two outputs are concatenated and passed to a
final linear prediction layer.

### Why It Matters

DCN showed that explicit bounded-degree feature crosses can be learned with low
additional parameter cost. It is a useful bridge between Wide & Deep, DeepFM,
and later cross-feature architectures such as xDeepFM and DCN variants.

### Key Papers

- [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)
  by Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang.
