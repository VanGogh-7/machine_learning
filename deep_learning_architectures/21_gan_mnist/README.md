# Fully Connected GAN for MNIST

This project implements an educational fully connected GAN for MNIST image
generation in PyTorch.

## What a GAN Is

GAN means Generative Adversarial Network. It contains two neural networks:

- Generator: maps random noise vectors to fake images.
- Discriminator: distinguishes real images from fake images.

The two networks are trained in opposition. The discriminator learns to classify
real and fake images correctly. The generator learns to fool the discriminator.

GANs are historically important because they introduced adversarial training for
generative modeling and helped prepare the path for DCGAN, Conditional GANs, and
later image generation systems.

## Architecture

This project uses the original simple MLP-style GAN, not DCGAN.

Key ideas:

- Random noise vector: sampled from a standard normal distribution.
- Generator loss: fake images are labeled as real because the generator wants to
  fool the discriminator.
- Discriminator loss: real images are labeled as 1 and fake images as 0.
- Fake image detach: fake images are detached when training the discriminator so
  discriminator updates do not change generator parameters.
- `BCEWithLogitsLoss`: combines sigmoid and binary cross entropy in a numerically
  stable way.
- Discriminator logits: the discriminator returns raw logits and does not apply
  sigmoid inside `forward`.
- Tanh output: generator output is in `[-1, 1]`, matching MNIST images normalized
  with `Normalize((0.5,), (0.5,))`.

GAN loss is useful for training, but it is not always a reliable image quality
metric. Generated samples should be inspected visually.

## Dataset

Dataset: MNIST.

Shared dataset location:

```text
machine_learning/datasets/mnist/
```

Only the official MNIST training split is used for GAN training.

## Tensor Shapes

Real image shape:

```text
[batch_size, 1, 28, 28]
```

Flattened real image shape:

```text
[batch_size, 784]
```

Noise vector shape:

```text
[batch_size, latent_dim]
```

Generated image shape:

```text
[batch_size, 1, 28, 28]
```

Discriminator output shape:

```text
[batch_size]
```

## Training

Run from this project directory:

```bash
python scripts/train.py
```

The script saves the latest generator and discriminator checkpoints after each
epoch and periodically writes generated sample grids under `outputs/`.

## Prediction

Run:

```bash
python scripts/predict.py
```

The script loads the generator checkpoint, samples random noise, generates
MNIST-like images, and saves image grids under `outputs/`.

This project implements an educational GAN, not a production image generation
system.

## Model Summary and Key Papers

### Historical Background

GAN introduced adversarial training for generative modeling. Instead of optimizing only reconstruction or likelihood-style losses, it trains two networks against each other.

### Basic Structure

The generator maps random noise to fake images. The discriminator receives real and fake images and returns logits indicating whether each image appears real.

### Why It Matters

GANs reshaped generative modeling research and led to many influential image generation architectures, including DCGAN, conditional GANs, and later high-resolution synthesis systems.

### Key Papers

* [Generative Adversarial Nets](https://proceedings.neurips.cc/paper_files/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html)
