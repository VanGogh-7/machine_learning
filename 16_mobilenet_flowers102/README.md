# MobileNet Flowers102

This project implements an educational MobileNet v1 style convolutional neural
network for Oxford 102 Flowers image classification.

## MobileNet

MobileNet introduced a practical lightweight CNN design based on depthwise
separable convolution. Earlier CNNs such as AlexNet, VGG, Inception, ResNet,
and DenseNet focused mainly on accuracy and representation power. MobileNet
focuses on efficient inference for mobile and resource-constrained devices.

Key ideas:

- **Standard convolution:** jointly mixes spatial and channel information.
- **Depthwise convolution:** applies one spatial filter per input channel.
- **Pointwise convolution:** uses a 1x1 convolution to mix channels.
- **Depthwise separable convolution:** combines depthwise and pointwise
  convolutions to reduce computation.
- **Width multiplier:** scales channel counts to trade accuracy for speed.
- **Adaptive average pooling:** turns final feature maps into a compact vector
  before classification.

Compared with VGG, MobileNet replaces most standard convolutions with cheaper
depthwise separable convolutions. Compared with Inception, it is not a
multi-branch design. Compared with ResNet, it does not rely on residual
connections. Compared with DenseNet, it does not concatenate features from all
previous layers.

This is an educational MobileNet v1 style model, not a production mobile
deployment pipeline.

## Dataset

The project uses Oxford 102 Flowers with 102 flower categories. The dataset is
stored centrally at:

```text
machine_learning/datasets/flowers102/
```

The official torchvision splits are used:

- train split
- validation split
- test split

The validation split is used for model selection. The test split is evaluated
only once after loading the best validation-accuracy checkpoint.

## Tensor Shapes

- Input images: `[batch_size, 3, image_size, image_size]`
- Output logits: `[batch_size, 102]`

## Project Structure

```text
16_mobilenet_flowers102/
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

MobileNet v1 was designed for mobile and resource-constrained vision applications. It focused on efficient CNN computation rather than only maximizing accuracy.

### Basic Structure

The core operation is depthwise separable convolution: a depthwise convolution applies one spatial filter per input channel, and a pointwise 1x1 convolution mixes channels. A width multiplier scales the model size.

### Why It Matters

MobileNet made efficient neural network design a major research direction for deployment on phones, embedded devices, and low-compute environments.

### Key Papers

* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
