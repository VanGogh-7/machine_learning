# Convolutional VAE for MNIST

This project implements an educational convolutional Variational AutoEncoder,
VAE, for MNIST image reconstruction and generation in PyTorch.

## What a VAE Is

A Variational AutoEncoder turns the AutoEncoder idea into a probabilistic
generative model. A standard AutoEncoder learns a deterministic latent vector.
A VAE learns a latent distribution by predicting:

- `mu`: the mean of the latent distribution
- `logvar`: the log variance of the latent distribution

The model samples a latent vector using the reparameterization trick and then
decodes that sample back into an image.

VAE is historically important because it combines representation learning with
probabilistic generation. It is a direct stepping stone toward later generative
models such as GAN, DCGAN, and diffusion models.

## Architecture

The model is implemented manually in PyTorch.

Key ideas:

- Encoder: convolutional layers downsample the image and extract features.
- Latent distribution: linear layers predict `mu` and `logvar`.
- Reparameterization trick: samples `z = mu + eps * std` while preserving
  differentiability.
- Decoder: a linear layer and transposed convolutions reconstruct the image.
- Reconstruction loss: binary cross entropy compares reconstructed images with
  originals.
- KL divergence: regularizes the latent distribution toward a standard normal
  distribution.
- Beta parameter: scales the KL term in the total VAE loss.

Because the latent space is regularized, the decoder can generate new digit-like
samples from random latent vectors sampled from a standard normal distribution.

## Dataset

Dataset: MNIST.

Shared dataset location:

```text
machine_learning/datasets/mnist/
```

The project uses:

- official MNIST training split
- validation split created from the training split
- official MNIST test split

The validation set is used for model selection. The test set is evaluated only
once after loading the best checkpoint.

## Tensor Shapes

Input image shape:

```text
[batch_size, 1, 28, 28]
```

Latent distribution shapes:

```text
mu: [batch_size, latent_dim]
logvar: [batch_size, latent_dim]
z: [batch_size, latent_dim]
```

Output reconstruction shape:

```text
[batch_size, 1, 28, 28]
```

## Training

Run from this project directory:

```bash
python scripts/train.py
```

The best checkpoint is saved based on validation total loss.

## Prediction

Run:

```bash
python scripts/predict.py
```

The script loads the best checkpoint, reconstructs MNIST test images, generates
new samples from random latent vectors, and saves figures under `outputs/`.

This project implements an educational convolutional VAE, not a production
generative model.
