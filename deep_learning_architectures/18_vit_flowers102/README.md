# Vision Transformer for Oxford 102 Flowers

This project implements a small educational Vision Transformer, ViT, for image
classification on the Oxford 102 Flowers dataset.

## What ViT Is

Vision Transformer applies the Transformer Encoder architecture to images.
Instead of using convolution as the main feature extractor, ViT splits an image
into fixed-size patches, projects each patch into an embedding vector, adds a
learnable positional embedding, and processes the resulting token sequence with
self-attention.

ViT is historically important because it showed that a pure Transformer-style
architecture can work well for image classification, especially when trained
with large-scale data. It changed the comparison point for vision models:

- CNNs use local convolutional inductive bias.
- MobileNet uses efficient depthwise separable convolution.
- ViT represents image patches as tokens and uses a Transformer Encoder.

## Architecture

The model is implemented manually in PyTorch and does not use pretrained ViT
models, `torchvision.models.vit`, `timm`, or `transformers`.

Key ideas:

- Image patches: the input image is split into non-overlapping patches.
- Patch embedding: a strided `Conv2d` projects patches into token embeddings.
- Image patches as tokens: each embedded patch becomes one Transformer token.
- Class token: a learnable token is prepended and used for final classification.
- Positional embedding: learnable position vectors are added so the model knows
  where each patch came from.
- Transformer Encoder: multi-head self-attention models relationships between
  all patches.
- MLP block: each Transformer layer includes a feed-forward network after
  attention.

## Dataset

Dataset: Oxford 102 Flowers.

Shared dataset location:

```text
machine_learning/datasets/flowers102/
```

The project uses the official torchvision splits:

- train split for training
- validation split for model selection
- test split only once after training

## Tensor Shapes

Input images:

```text
[batch_size, 3, image_size, image_size]
```

Patch token shape:

```text
[batch_size, num_patches, embedding_dim]
```

Token sequence after adding the class token:

```text
[batch_size, num_patches + 1, embedding_dim]
```

Model output:

```text
[batch_size, 102]
```

With the default configuration:

- `image_size = 128`
- `patch_size = 16`
- `num_patches = 64`
- `embedding_dim = 192`

## Training

Run from this project directory:

```bash
python scripts/train.py
```

The script prints dataset sizes, one batch shape, and model output shape before
training. The best checkpoint is saved based on validation accuracy.

## Prediction

Run:

```bash
python scripts/predict.py
```

The script loads the best checkpoint, predicts several test images, prints true
and predicted labels, and saves a prediction grid under `outputs/`.

This is an educational small ViT from scratch, not a pretrained ViT or a
production image-classification system.

## Model Summary and Key Papers

### Historical Background

Vision Transformer, ViT, introduced a pure Transformer-style architecture for image classification. It showed that images can be processed as sequences of patch tokens.

### Basic Structure

An image is split into fixed-size patches, each patch is embedded as a token, a class token and positional embeddings are added, and a Transformer Encoder processes the token sequence.

### Why It Matters

ViT challenged the assumption that convolution is required for strong vision models and helped connect computer vision with the Transformer scaling trends from NLP.

### Key Papers

* [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy)
