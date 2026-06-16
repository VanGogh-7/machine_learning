# ResNet CIFAR-10

This PyTorch project trains a small ResNet-style convolutional neural network
to classify images from the CIFAR-10 dataset.

## ResNet

ResNet introduced residual connections, which allow a block to learn changes
to its input instead of learning an entirely new representation. The shortcut
adds the block input to the convolution output, helping gradients flow through
deep networks and making very deep models easier to optimize.

ResNet was historically important because it enabled substantially deeper
networks while improving accuracy. The original ImageNet ResNet-50 is much
larger than necessary for 32 x 32 CIFAR-10 images, so this project uses a
smaller manually implemented ResNet with six residual blocks.

## Architecture Shapes

- Input: `[batch_size, 3, 32, 32]`
- Initial convolution: `[batch_size, 64, 32, 32]`
- Residual stage 1: `[batch_size, 64, 32, 32]`
- Residual stage 2: `[batch_size, 128, 16, 16]`
- Residual stage 3: `[batch_size, 256, 8, 8]`
- Adaptive average pooling: `[batch_size, 256, 1, 1]`
- Output logits: `[batch_size, 10]`

Projection shortcuts use a 1 x 1 convolution when a residual block changes
the number of channels or spatial dimensions.

## Dataset

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
06_resnet_cifar10/
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
validation accuracy as `resnet_cifar10.pt`, then loads it and evaluates the
test set once.

## Predict

After training:

```bash
python scripts/predict.py
```

The prediction script loads the best checkpoint and prints the true and
predicted label indices and class names for one CIFAR-10 test image.

## Model Summary and Key Papers

### Historical Background

ResNet introduced residual learning to make very deep networks easier to optimize. Instead of forcing each block to learn a full transformation, residual blocks learn a correction added to the input through a skip connection.

### Basic Structure

The core unit is a residual block with convolution, normalization, activation, and an identity or projection shortcut. This project implements a smaller CIFAR-10 adapted ResNet manually.

### Why It Matters

Residual connections made extremely deep CNNs practical and influenced many later architectures in vision, language, speech, and generative modeling.

### Key Papers

* [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
* [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
