# Wide & Deep CTR Prediction

This project implements an educational Wide & Deep model for click-through-rate
(CTR) prediction using PyTorch.

## Wide & Deep

Wide & Deep combines memorization and generalization in one ranking model:

- The **wide component** is a linear memorization part. It learns direct effects
  from numerical features and sparse categorical IDs.
- The **deep component** embeds sparse categorical features and combines them
  with numerical features in a multilayer perceptron.
- The final prediction adds the wide logit and the deep logit, so the model can
  use both memorized sparse patterns and generalized dense interactions.

Wide & Deep is historically important in recommender systems, advertising
systems, ranking systems, and CTR prediction because it showed how a linear
memorization model and a deep generalization model can be jointly trained for
large-scale recommendation.

CTR prediction estimates whether a user will click an item or advertisement.
The label is binary: `0` means not clicked and `1` means clicked. Numerical
features usually represent continuous or count-based signals. Sparse categorical
features represent high-cardinality fields such as user, item, ad, context, or
category IDs.

The model returns raw logits instead of probabilities. Training uses
`BCEWithLogitsLoss`, which combines sigmoid and binary cross-entropy in a
numerically stable operation. Probabilities are computed with sigmoid only for
metrics and prediction output.

This model is a natural starting point before DeepFM, DCN, xDeepFM, DIN, DIEN,
and other recommender-system architectures.

## Dataset

Use a real Criteo-style CTR dataset stored centrally at:

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
- Raw logits: `[batch_size]`
- Probabilities after sigmoid: `[batch_size]`

## Project Structure

```text
01_wide_deep_ctr_prediction/
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

Wide & Deep was introduced for large-scale recommendation systems and evaluated
in an industrial app recommendation setting. It is an early and influential
hybrid neural recommender architecture.

### Basic Structure

The wide part is a linear model for memorization. The deep part uses categorical
embeddings and nonlinear layers for generalization. Their scalar logits are
added before the binary CTR loss.

### Why It Matters

Wide & Deep helped popularize jointly trained hybrid recommender models. It
made the memorization-versus-generalization tradeoff explicit and remains a
useful baseline for CTR ranking projects.

### Key Papers

- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
  by Heng-Tze Cheng et al.
