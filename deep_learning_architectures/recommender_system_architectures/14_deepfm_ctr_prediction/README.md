# DeepFM CTR Prediction

This project implements an educational DeepFM model for click-through-rate
(CTR) prediction using PyTorch.

## DeepFM

DeepFM combines a Factorization Machine (FM) component with a deep neural
network in one end-to-end model:

- **First-order terms** model the individual effects of numerical and sparse
  categorical features.
- **Second-order FM interactions** efficiently model pairwise interactions
  between categorical fields through embedding vectors.
- **The deep component** learns higher-order nonlinear interactions from the
  same categorical embeddings and numerical features.

DeepFM is historically important in recommender systems and advertising
because it learns useful feature interactions automatically. Compared with
Wide & Deep, it reduces the need to manually define feature crosses in a wide
component. It is a natural next step after Wide & Deep and before DCN, DIN,
and other recommender-system architectures.

CTR prediction estimates whether a user will click an item or advertisement.
The model returns raw logits because `BCEWithLogitsLoss` combines a numerically
stable sigmoid operation with binary cross-entropy.

## Dataset

DeepFM reuses the same real processed Criteo-style dataset as the Wide & Deep
project:

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

If all three files exist, the provided splits are used directly. If exactly
one processed CSV exists, it is deterministically split into 80% train, 10%
validation, and 10% test subsets in memory. Missing data produces a clear
error; the project does not create a toy dataset.

All numerical normalization statistics and categorical vocabularies are fitted
from training data only. Validation AUC selects the best checkpoint. The test
split is evaluated only once after training.

## Tensor Shapes

- Numerical features: `[batch_size, num_numerical_features]`
- Categorical features: `[batch_size, num_categorical_features]`
- Raw logits: `[batch_size]`
- Probabilities after sigmoid: `[batch_size]`

## Project Structure

```text
14_deepfm_ctr_prediction/
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
