# AlexNet CIFAR-10

This PyTorch project trains an AlexNet-style convolutional neural network to
classify images from the CIFAR-10 dataset.

## AlexNet

AlexNet won the 2012 ImageNet competition by a large margin and helped
establish deep convolutional neural networks as the standard approach for
large-scale image classification. Its use of ReLU activations, dropout, data
augmentation, and GPU training strongly influenced later CNN architectures.

The original AlexNet was designed for much larger ImageNet images and used
large convolution kernels and very large fully connected layers. CIFAR-10
images are only 32 x 32 pixels, so this project uses smaller 3 x 3
convolutions and a reduced classifier while preserving the general AlexNet
pattern of stacked convolutions, max pooling, ReLU activations, and dropout.

## Dataset

CIFAR-10 contains 50,000 training images and 10,000 test images across 10
classes. Each image is an RGB image with shape `[3, 32, 32]`.

Datasets are shared across the repository. CIFAR-10 is stored under:

```text
machine_learning/datasets/cifar10/
```

The training transform uses random cropping and horizontal flipping. Training
and test images are normalized with the CIFAR-10 channel mean and standard
deviation.

## Project Structure

```text
04_alexnet_cifar10/
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

## Tensor Shapes

- Input: `[batch_size, 3, 32, 32]`
- Feature output after three pooling layers: `[batch_size, 256, 4, 4]`
- Flattened features: `[batch_size, 4096]`
- Output logits: `[batch_size, 10]`

## Setup

From the project root:

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py
```

Training automatically selects CUDA, MPS, or CPU, prints the first batch
shapes, and saves the checkpoint with the best test accuracy as
`alexnet_cifar10.pt`.

## Predict

After training:

```bash
python scripts/predict.py
```

The prediction script loads the saved checkpoint and prints the true and
predicted labels and class names for one CIFAR-10 test image.

## Model Summary and Key Papers

### Historical Background

AlexNet was the breakthrough CNN that won ImageNet 2012 by a large margin and helped trigger the modern deep learning wave. It demonstrated that deep CNNs could scale to large visual recognition tasks when combined with GPUs, ReLU activations, data augmentation, and dropout.

### Basic Structure

The original AlexNet stacks convolutional layers, ReLU activations, pooling, dropout, and fully connected classifier layers. This project adapts the idea to CIFAR-10 with a smaller CNN because CIFAR-10 images are only 32x32.

### Why It Matters

AlexNet made deep convolutional networks the dominant approach for computer vision and changed the direction of visual recognition research.

### Key Papers

* [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
