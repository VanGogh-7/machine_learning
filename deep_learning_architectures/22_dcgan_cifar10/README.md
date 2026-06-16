# DCGAN for CIFAR-10

This project implements an educational Deep Convolutional GAN, DCGAN, for
CIFAR-10 image generation in PyTorch.

## What DCGAN Is

DCGAN is a convolutional version of GAN. It keeps the adversarial setup of a
generator and discriminator, but replaces fully connected image generation with
convolutional networks designed for images.

DCGAN is historically important because it popularized a practical and stable
GAN architecture:

- transposed convolutions in the generator
- strided convolutions in the discriminator
- BatchNorm in both networks
- ReLU in the generator except the output layer
- LeakyReLU in the discriminator
- Tanh generator output
- image normalization to `[-1, 1]`

## Difference From `21_gan_mnist`

The previous project, `21_gan_mnist`, uses fully connected networks. This
project uses convolutional generator and discriminator networks, which are more
appropriate for image structure.

## Training Idea

The generator maps a random noise tensor to a fake CIFAR-10-like image. The
discriminator receives real and fake images and returns raw logits.

Fake images are detached when training the discriminator so discriminator
updates do not change generator parameters. The generator is trained separately
to make fake images receive real labels from the discriminator.

`BCEWithLogitsLoss` is used because it combines sigmoid and binary cross entropy
in a numerically stable way. The discriminator does not apply sigmoid inside its
forward method.

GAN loss is useful for training, but it is not always a reliable image quality
metric. Generated samples should be inspected visually.

## Dataset

Dataset: CIFAR-10.

Shared dataset location:

```text
machine_learning/datasets/cifar10/
```

Only the official CIFAR-10 training split is used for DCGAN training.

## Tensor Shapes

Real image shape:

```text
[batch_size, 3, 32, 32]
```

Noise tensor shape:

```text
[batch_size, latent_dim, 1, 1]
```

Generated image shape:

```text
[batch_size, 3, 32, 32]
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
CIFAR-10-like images, and saves image grids under `outputs/`.

This project implements an educational DCGAN, not a production image generation
system.
