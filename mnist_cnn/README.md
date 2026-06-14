# MNIST CNN

This beginner-friendly PyTorch project trains a simple convolutional neural
network (CNN) to classify handwritten digits from the MNIST dataset.

## Project structure

```text
mnist_cnn/
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

`src/` contains reusable project code. `scripts/` contains the training and
prediction entry points.

## Shared dataset storage

MNIST is shared with the rest of the repository and stored under
`machine_learning/datasets/mnist/`. Running this project reuses that copy
instead of downloading a project-local duplicate.

## Setup

From the project root:

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py
```

Training downloads MNIST when it is not already available in the shared
dataset directory, displays the learning curves, and saves the model weights
as `mnist_cnn.pt`.

## Predict

After training:

```bash
python scripts/predict.py
```

This loads the saved model and prints predictions for a few validation images.
