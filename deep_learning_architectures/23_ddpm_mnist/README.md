# DDPM for MNIST

This project implements an educational Denoising Diffusion Probabilistic Model,
DDPM, for MNIST image generation in PyTorch.

## What DDPM Is

DDPM is a diffusion-based generative model. It learns image generation through
iterative denoising instead of adversarial training.

The forward diffusion process gradually adds Gaussian noise to real images. The
reverse process starts from pure Gaussian noise and repeatedly denoises it until
an image is produced.

Diffusion models are historically important because they became one of the main
foundations of modern image generation systems.

## Key Ideas

- Beta schedule: controls how much noise is added at each timestep.
- Alpha: `1 - beta`.
- Alpha bar: cumulative product of alphas used to sample noisy images directly.
- Forward diffusion: creates `x_t` from a clean image and Gaussian noise.
- Reverse denoising: starts from noise and removes noise step by step.
- Noise prediction objective: the U-Net predicts the noise that was added.
- Timestep embedding: tells the model how noisy the current image is.
- U-Net noise predictor: combines encoder context, decoder upsampling, and skip
  connections.
- MSE loss: compares predicted noise with the true sampled noise.

Compared with a VAE, DDPM does not learn one compact latent vector. Compared with
a GAN, DDPM does not use a discriminator or adversarial loss.

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

Real image shape:

```text
[batch_size, 1, 28, 28]
```

Noisy image shape:

```text
[batch_size, 1, 28, 28]
```

Timestep shape:

```text
[batch_size]
```

Predicted noise shape:

```text
[batch_size, 1, 28, 28]
```

## Training

Run from this project directory:

```bash
python scripts/train.py
```

The best checkpoint is saved based on validation noise prediction loss.

## Prediction

Run:

```bash
python scripts/predict.py
```

The script loads the best checkpoint, generates MNIST-like samples from Gaussian
noise, and saves generated images plus selected reverse-denoising steps under
`outputs/`.

This project implements an educational DDPM, not a production-scale diffusion
model.

## Model Summary and Key Papers

### Historical Background

DDPM is a diffusion-based generative model that learns generation through iterative denoising. It became one of the foundations of modern diffusion image generation systems.

### Basic Structure

The forward process gradually adds Gaussian noise using a beta schedule. The reverse process uses a timestep-conditioned U-Net to predict noise and denoise images step by step.

### Why It Matters

DDPM showed that high-quality generation can be trained with a simple noise prediction objective rather than adversarial training, strongly influencing modern generative modeling.

### Key Papers

* [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
* [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)
* [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
