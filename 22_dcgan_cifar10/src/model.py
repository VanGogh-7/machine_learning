import torch
from torch import nn


class DCGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        image_channels: int = 3,
        feature_maps: int = 64,
    ) -> None:
        super().__init__()
        # Transposed convolutions progressively upsample noise from 1x1 to 32x32.
        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim,
                feature_maps * 4,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                feature_maps * 4,
                feature_maps * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                feature_maps * 2,
                feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                feature_maps,
                image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # Tanh matches real CIFAR-10 images normalized to [-1, 1].
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z is a random noise tensor with shape [batch_size, latent_dim, 1, 1].
        return self.network(z)


class DCGANDiscriminator(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        feature_maps: int = 64,
    ) -> None:
        super().__init__()
        # Strided convolutions downsample images from 32x32 to a single logit.
        self.network = nn.Sequential(
            nn.Conv2d(
                image_channels,
                feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                feature_maps,
                feature_maps * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                feature_maps * 2,
                feature_maps * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                feature_maps * 4,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Return raw logits. BCEWithLogitsLoss applies sigmoid internally.
        logits = self.network(images)
        return logits.view(images.size(0))


def weights_init_dcgan(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


if __name__ == "__main__":
    z = torch.randn(4, 100, 1, 1)
    generator = DCGANGenerator(latent_dim=100, image_channels=3, feature_maps=64)
    fake_images = generator(z)

    discriminator = DCGANDiscriminator(image_channels=3, feature_maps=64)
    logits = discriminator(fake_images)

    print(fake_images.shape)
    print(logits.shape)

