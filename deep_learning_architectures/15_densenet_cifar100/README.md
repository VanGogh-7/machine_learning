# DenseNet CIFAR-100

This project implements an educational DenseNet-style convolutional neural
network for CIFAR-100 classification.

## DenseNet

DenseNet introduced dense connections between layers. Inside a dense block,
each layer receives all previous feature maps as input and concatenates its new
feature maps to the block output. This feature concatenation encourages feature
reuse and improves gradient flow through the network.

Key ideas:

- **Dense connection:** every layer sees earlier features directly.
- **Feature concatenation:** DenseNet concatenates channels instead of adding
  tensors as ResNet does.
- **Growth rate:** each dense layer adds a fixed number of new feature maps.
- **Bottleneck layer:** a 1x1 convolution reduces computation before the 3x3
  convolution.
- **Transition layer:** a 1x1 convolution plus average pooling reduces channel
  count and spatial resolution between dense blocks.
- **Compression factor:** controls how strongly transition layers reduce
  channels.

Compared with VGG, DenseNet reuses features instead of only stacking plain
convolutions. Compared with Inception, it focuses on dense feature reuse rather
than parallel multi-scale branches. Compared with ResNet, it uses channel
concatenation rather than residual addition.

This is a CIFAR-sized educational DenseNet, not necessarily the exact full
ImageNet DenseNet variant.

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
15_densenet_cifar100/
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
