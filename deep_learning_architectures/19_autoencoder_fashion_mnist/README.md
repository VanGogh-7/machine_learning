# Convolutional AutoEncoder for FashionMNIST

This project implements an educational convolutional AutoEncoder for
FashionMNIST image reconstruction in PyTorch.

## What an AutoEncoder Is

An AutoEncoder is a representation learning architecture that learns to
reconstruct its input. The encoder compresses an input image into a latent
representation, and the decoder reconstructs the original image from that
compressed representation.

AutoEncoders are historically important because they are classic unsupervised
or self-supervised representation learning models. They prepare the ground for
later generative models such as Variational AutoEncoders, GANs, DCGANs, and
diffusion models.

## Architecture

The model is implemented manually in PyTorch.

Key ideas:

- Encoder: convolutional layers downsample the image and extract features.
- Bottleneck: a linear layer maps features to a compact latent vector.
- Latent representation: compressed image information with shape
  `[batch_size, latent_dim]`.
- Decoder: a linear layer and transposed convolutions reconstruct the image.
- Reconstruction: the target is the input image itself.
- Reconstruction loss: `MSELoss` compares reconstructed images with originals.
- Output sigmoid: keeps reconstructed pixel values in `[0, 1]`, matching
  `ToTensor()` inputs.

Unlike a classifier, this model does not predict a class label. It learns a
compressed representation that can recreate the input image.

## Dataset

Dataset: FashionMNIST.

Shared dataset location:

```text
machine_learning/datasets/fashion_mnist/
```

The project uses:

- official FashionMNIST training split
- validation split created from the training split
- official FashionMNIST test split

The validation set is used for model selection. The test set is evaluated only
once after loading the best checkpoint.

## Tensor Shapes

Input image shape:

```text
[batch_size, 1, 28, 28]
```

Latent representation shape:

```text
[batch_size, latent_dim]
```

Reconstruction output shape:

```text
[batch_size, 1, 28, 28]
```

## Training

Run from this project directory:

```bash
python scripts/train.py
```

The best checkpoint is saved based on validation reconstruction loss.

## Prediction

Run:

```bash
python scripts/predict.py
```

The script loads the best checkpoint, reconstructs several FashionMNIST test
images, prints the selected batch reconstruction loss, and saves an original vs.
reconstruction comparison figure under `outputs/`.

This project implements an educational convolutional AutoEncoder, not a
production image compression system.
