# U-Net Pet Segmentation

This project implements an educational U-Net for semantic segmentation on the
Oxford-IIIT Pet dataset.

## U-Net

U-Net is a classic encoder-decoder architecture for image segmentation. The
encoder path captures context by reducing spatial resolution. The decoder path
upsamples features back to the original resolution. Skip connections copy
high-resolution encoder features into the decoder, helping recover precise
segmentation boundaries.

Semantic segmentation is pixel-level classification. Unlike image
classification CNNs, which output one class label for the whole image, U-Net
outputs a class label for every pixel.

Key ideas:

- Encoder path for context.
- Decoder path for spatial recovery.
- Skip connections between matching resolutions.
- Feature concatenation to preserve fine details.
- Upsampling with transposed convolutions.
- A final 1x1 convolution mapping features to segmentation classes.

This is an educational U-Net, not a production segmentation system.

## Dataset

The project uses Oxford-IIIT Pet segmentation and stores it centrally at:

```text
machine_learning/datasets/oxford_iiit_pet/
```

The torchvision segmentation target is a trimap. This project converts it to
three semantic classes:

- `0`: background
- `1`: pet
- `2`: border

The `trainval` split is divided deterministically into training and validation
sets. The official test split is evaluated only once after loading the best
validation mean-IoU checkpoint.

## Tensor Shapes

- Input images: `[batch_size, 3, image_size, image_size]`
- Target masks: `[batch_size, image_size, image_size]`
- Output logits: `[batch_size, num_classes, image_size, image_size]`

Masks are `LongTensor` class IDs and are not one-hot encoded.

## Project Structure

```text
17_unet_pet_segmentation/
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

Prediction saves comparison figures with input image, ground-truth mask, and
predicted mask under the project output directory.
