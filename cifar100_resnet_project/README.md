# CIFAR-100 Small ResNet

This project trains a small, manually implemented ResNet-style image classifier with PyTorch. It
uses CIFAR-100 because its RGB images and 100 classes provide a practical introduction to data
augmentation, residual connections, regularization, learning-rate scheduling, checkpointing, and
top-k evaluation.

## Project structure

```text
cifar100_resnet_project/
├── checkpoints/       # Best model checkpoint
├── scripts/           # Training, evaluation, and prediction entry points
└── src/               # Config, data, models, training engine, metrics, and plotting
```

`SmallCNN` provides a conventional CNN baseline. `SmallResNet` is the default model and implements
its residual blocks from scratch without pretrained weights.

## Shared dataset storage

CIFAR-100 is shared with the rest of the repository and stored under
`machine_learning/datasets/cifar100/`. Running this project reuses that copy instead of downloading
a project-local duplicate.

## Setup and usage

Run these commands from the project root:

```bash
pip install -r requirements.txt
python scripts/train.py
python scripts/evaluate.py
python scripts/predict.py
```

The first command that accesses CIFAR-100 downloads it into the shared dataset directory. Training
saves the model with the best validation top-1 accuracy to `checkpoints/best_model.pt`.
