# Inception CIFAR-100

This project implements an educational GoogLeNet / Inception v1 style
convolutional neural network for CIFAR-100 classification.

## Inception

GoogLeNet, also called Inception v1, introduced the Inception module. Instead
of using one convolution path, an Inception module applies several branches in
parallel and concatenates their outputs. This lets the network extract
multi-scale features with 1x1, 3x3, 5x5, and pooling-based branches.

The 1x1 convolutions act as bottlenecks before the larger 3x3 and 5x5
convolutions, reducing computation while keeping useful channel mixing.
Compared with VGG, Inception is less uniform and uses multi-branch feature
extraction instead of simply stacking many 3x3 convolutions. Compared with
ResNet, Inception focuses on parallel feature branches rather than residual
skip connections.

This is a CIFAR-sized educational model, not the full ImageNet GoogLeNet
architecture exactly.

## Dataset

The project uses CIFAR-100 with 100 classes. The dataset is stored centrally at:

```text
machine_learning/datasets/cifar100/
```

The official CIFAR-100 training split is divided deterministically into:

- 90% training data
- 10% validation data

The official CIFAR-100 test split is evaluated only once after loading the best
validation-accuracy checkpoint.

## Tensor Shapes

- Input images: `[batch_size, 3, 32, 32]`
- Output logits: `[batch_size, 100]`

## Project Structure

```text
14_inception_cifar100/
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

## Model Summary and Key Papers

### Historical Background

GoogLeNet, also known as Inception v1, introduced Inception modules for multi-branch convolutional feature extraction. It showed that CNNs could be made deeper and more efficient with carefully designed modules.

### Basic Structure

An Inception block applies parallel 1x1, 3x3, and 5x5 convolution branches plus a pooling branch. 1x1 convolutions act as bottlenecks to reduce computation before larger convolutions.

### Why It Matters

Inception made multi-scale feature extraction and efficient bottleneck design central ideas in CNN architecture research.

### Key Papers

* [Going Deeper with Convolutions](https://openaccess.thecvf.com/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)
