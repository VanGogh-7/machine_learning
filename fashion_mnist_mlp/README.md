# FashionMNIST MLP

This beginner-friendly PyTorch project trains a simple multilayer perceptron
(MLP) to classify clothing images from the FashionMNIST dataset.

## Project structure

```text
fashion_mnist_mlp/
├── README.md
├── requirements.txt
├── scripts/
│   ├── train.py
│   └── predict.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── model.py
    ├── engine.py
    ├── utils.py
    └── visualize.py
```

`src/` contains the reusable project code. `scripts/` contains the commands
used to train the model and run inference.

## Shared dataset storage

Fashion-MNIST is shared with the rest of the repository and stored under
`machine_learning/datasets/fashion_mnist/`. Running this project reuses that
copy instead of downloading a project-local duplicate.

## Setup

From the project root:

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py
```

Training downloads FashionMNIST when it is not already available in the shared
dataset directory, displays the learning curves, and saves the model weights
as `fashion_mnist_mlp.pt`.

## Predict

After training:

```bash
python scripts/predict.py
```

This loads the saved model and prints predictions for a few validation images.
