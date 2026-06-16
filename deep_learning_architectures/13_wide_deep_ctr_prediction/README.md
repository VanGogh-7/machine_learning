# Wide & Deep CTR Prediction

This project implements an educational Wide & Deep model for click-through-rate
(CTR) prediction using PyTorch.

## Wide & Deep

Wide & Deep combines memorization and generalization:

- The **wide component** memorizes linear patterns in numerical features and
  sparse categorical feature IDs.
- The **deep component** embeds categorical fields and learns nonlinear feature
  interactions with a multilayer perceptron.
- The two scalar logits are added before computing the click probability.

Wide & Deep was historically important in industrial recommender and
advertising systems. It is a natural starting point before architectures such
as DeepFM, DCN, and DIN.

CTR prediction estimates whether a user will click an item or advertisement.
Numerical features represent continuous or count-based information. Sparse
categorical features represent fields such as user, item, context, or category.
The model returns raw logits because `BCEWithLogitsLoss` combines a numerically
stable sigmoid operation with binary cross-entropy.

## Dataset

Use a real processed Criteo-style CTR dataset stored centrally at:

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

If all three files exist, the project uses the provided splits. If exactly one
of these processed CSVs exists, it is deterministically split into 80% train,
10% validation, and 10% test subsets in memory. Missing data produces a clear
error; the project does not create a toy dataset.

All numerical normalization statistics and categorical vocabularies are fitted
from the training split only. Validation AUC selects the best checkpoint. The
test split is evaluated only once after training.

## Tensor Shapes

- Numerical features: `[batch_size, num_numerical_features]`
- Categorical features: `[batch_size, num_categorical_features]`
- Raw logits: `[batch_size]`
- Probabilities after sigmoid: `[batch_size]`

## Project Structure

```text
13_wide_deep_ctr_prediction/
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

## Train

```bash
python scripts/train.py
```

## Predict

After training:

```bash
python scripts/predict.py
```

This project is an educational PyTorch implementation, not a full production
CTR system.

## Model Summary and Key Papers

### Historical Background

Wide & Deep is an important industrial recommender-system architecture that combines memorization and generalization. It was designed for large-scale recommendation and click-through-rate style prediction tasks.

### Basic Structure

The wide component is a linear model that can memorize sparse feature patterns. The deep component uses embeddings and neural layers to learn nonlinear feature interactions.

### Why It Matters

Wide & Deep helped popularize hybrid recommender models that combine manually useful feature patterns with learned dense representations.

### Key Papers

* [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
