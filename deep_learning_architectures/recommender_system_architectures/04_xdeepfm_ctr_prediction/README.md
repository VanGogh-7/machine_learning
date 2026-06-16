# xDeepFM CTR Prediction

This project implements an educational xDeepFM model for click-through-rate
(CTR) prediction using PyTorch.

## xDeepFM

xDeepFM combines linear memorization, explicit vector-wise feature interaction
learning, and implicit deep feature interaction learning:

- The **linear component** learns first-order effects from sparse categorical
  feature IDs and numerical features.
- The **embedding layer** maps each sparse categorical field to a dense vector.
- The **Compressed Interaction Network (CIN)** explicitly models vector-wise
  high-order feature interactions.
- The **deep component** learns implicit nonlinear interactions with an MLP.
- The final CTR logit adds the linear logit, CIN logit, and deep logit.

xDeepFM is historically important in recommender systems, advertising systems,
and CTR prediction because it learns both explicit and implicit feature
interactions without manual feature crossing. Compared with Wide & Deep, it
reduces the need for hand-designed cross features. Compared with DeepFM, it
models explicit high-order feature interactions through CIN instead of relying
only on the FM second-order interaction formula. Compared with DCN, xDeepFM
focuses on vector-wise feature interactions instead of scalar feature crossing.

CTR prediction estimates whether a user will click an item or advertisement.
The label is binary: `0` means not clicked and `1` means clicked. Numerical
features usually represent continuous or count-based signals. Sparse categorical
features represent high-cardinality fields such as user, item, ad, context, or
category IDs.

In this implementation, CIN starts from:

```text
field_embeddings: [batch_size, num_categorical_fields, embedding_dim]
```

At each CIN layer, the previous layer output interacts with the original field
embeddings. The interaction tensor is compressed with a `Conv1d` layer and
pooled across the embedding dimension. The concatenated pooled outputs form:

```text
cin_output: [batch_size, sum(cin_layer_sizes)]
```

The model returns raw logits instead of probabilities. Training uses
`BCEWithLogitsLoss`, which combines sigmoid and binary cross-entropy in a
numerically stable operation. Probabilities are computed with sigmoid only for
metrics and prediction output.

xDeepFM is a natural next step after Wide & Deep, DeepFM, and DCN, and before
DIN, DIEN, and other behavior-sequence recommendation models.

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
- Field embeddings: `[batch_size, num_categorical_fields, embedding_dim]`
- Deep input: `[batch_size, num_categorical_fields * embedding_dim + num_numerical_features]`
- CIN output: `[batch_size, sum(cin_layer_sizes)]`
- Raw logits: `[batch_size]`
- Probabilities after sigmoid: `[batch_size]`

## Project Structure

```text
04_xdeepfm_ctr_prediction/
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

xDeepFM was introduced to combine explicit and implicit feature interaction
learning for recommender systems. Its CIN component was designed to generate
feature interactions explicitly at the vector-wise level.

### Basic Structure

xDeepFM uses a linear component for first-order effects, a CIN for explicit
high-order vector-wise interactions, and a deep network for implicit nonlinear
interactions. Their logits are added for the final CTR prediction.

### Why It Matters

xDeepFM is a useful bridge from DeepFM and DCN toward richer feature-interaction
models. It keeps the practical CTR setup while showing how explicit high-order
interactions can be modeled without manual feature engineering.

### Key Papers

- [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170)
  by Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and
  Guangzhong Sun.
