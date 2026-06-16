# VGG CIFAR-10

This PyTorch project trains a VGG-style convolutional neural network to
classify images from the CIFAR-10 dataset.

## VGGNet

VGGNet demonstrated that deep convolutional networks can be built from a
simple, uniform design: repeated small 3 x 3 convolutions followed by pooling.
This approach made network depth and feature extraction easier to understand
and strongly influenced later CNN architectures.

The original VGG-16 was designed for large ImageNet images and has a very
large classifier. This project uses a smaller VGG-style model adapted for
32 x 32 CIFAR-10 images. It preserves repeated 3 x 3 convolutions while using
three convolution blocks and a smaller classifier.

## Dataset

CIFAR-10 contains RGB images from 10 classes. The model input and output
shapes are:

- Input: `[batch_size, 3, 32, 32]`
- Output logits: `[batch_size, 10]`

The original CIFAR-10 training split is deterministically divided into:

- 45,000 training images
- 5,000 validation images

The validation set is used for model selection and saving the best checkpoint.
The separate 10,000-image test set is evaluated only once after training.

Datasets are shared across the repository. CIFAR-10 is stored under:

```text
machine_learning/datasets/cifar10/
```

## Project Structure

```text
05_vgg_cifar10/
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

## Setup

From the project root:

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py
```

Training prints dataset and tensor shapes, saves the checkpoint with the best
validation accuracy as `vgg_cifar10.pt`, then loads it and evaluates the test
set once.

## Predict

After training:

```bash
python scripts/predict.py
```

The prediction script loads the best checkpoint and prints the true and
predicted label indices and class names for one CIFAR-10 test image.
