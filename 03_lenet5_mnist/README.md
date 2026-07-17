# LeNet-5 MNIST

This PyTorch project trains LeNet-5 to classify handwritten digits from the
MNIST dataset.

## LeNet-5

LeNet-5 is an early convolutional neural network designed for handwritten
digit recognition. It alternates convolution and average-pooling layers to
extract spatial features, then uses fully connected layers for classification.
This implementation uses `Tanh` activations, matching the original
architecture, and pads the first convolution so 28 x 28 MNIST images produce
the classic 16 x 5 x 5 feature tensor.

## Dataset

MNIST contains 60,000 training images and 10,000 test images. Each grayscale
image is 28 x 28 pixels and belongs to one of 10 digit classes. The training
set is split into 55,000 training images and 5,000 validation images.

Images are converted with `transforms.ToTensor()` and normalized with:

```python
transforms.Normalize((0.1307,), (0.3081,))
```

MNIST is shared with the rest of the repository and stored under
`machine_learning/datasets/mnist/`. Running this project reuses that copy
instead of downloading a project-local duplicate.

## Project Structure

```text
03_lenet5_mnist/
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

The best validation checkpoint is created as `lenet5_mnist.pt` after training.

## Setup

From the project root:

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py
```

The training script automatically selects CUDA, MPS, or CPU, prints the first
batch shapes before training, plots learning curves, and saves the model with
the best validation accuracy.

## Predict

After training:

```bash
python scripts/predict.py
```

The prediction script loads `lenet5_mnist.pt` and predicts one sample from the
MNIST test set.

## Tensor Shapes

For a batch size of 64, tensors move through the network as follows:

| Stage | Tensor shape |
| --- | --- |
| Input batch | `(64, 1, 28, 28)` |
| Conv2d, 1 to 6 channels | `(64, 6, 28, 28)` |
| Average pooling | `(64, 6, 14, 14)` |
| Conv2d, 6 to 16 channels | `(64, 16, 10, 10)` |
| Average pooling | `(64, 16, 5, 5)` |
| Flatten | `(64, 400)` |
| Fully connected | `(64, 120)` |
| Fully connected | `(64, 84)` |
| Output logits | `(64, 10)` |

## Model Summary and Key Papers

### Historical Background

LeNet-5 is one of the earliest successful convolutional neural networks for handwritten digit recognition. It showed that learned convolutional filters, local receptive fields, and subsampling could work together for practical document recognition.

### Basic Structure

The model uses convolutional layers to extract local visual features, pooling or subsampling layers to reduce spatial size, and fully connected layers to classify the final representation.

### Why It Matters

LeNet-5 established many design ideas that still appear in modern CNNs: convolution, spatial downsampling, hierarchical feature learning, and end-to-end training.

### Key Papers

* [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
