import torch
from torch import nn


class ConvAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        base_channels: int = 32,
        image_size: int = 28,
    ) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4.")

        self.encoded_size = image_size // 4
        self.encoded_channels = base_channels * 2
        self.feature_dim = self.encoded_channels * self.encoded_size * self.encoded_size

        # The encoder compresses the input image into a compact latent vector.
        self.encoder_features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_channels,
                self.encoded_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.encoder_projection = nn.Linear(self.feature_dim, latent_dim)

        # The decoder expands the latent representation back to image space.
        self.decoder_projection = nn.Sequential(
            nn.Linear(latent_dim, self.feature_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder_features = nn.Sequential(
            nn.ConvTranspose2d(
                self.encoded_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels,
                in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            # FashionMNIST tensors from ToTensor are in [0, 1], so sigmoid
            # keeps reconstructions in the same range.
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder_features(images)
        return self.encoder_projection(features)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.decoder_projection(latent)
        features = features.view(
            latent.size(0),
            self.encoded_channels,
            self.encoded_size,
            self.encoded_size,
        )
        return self.decoder_features(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # The target for an autoencoder is the input image itself.
        latent = self.encode(images)
        return self.decode(latent)


if __name__ == "__main__":
    x = torch.randn(4, 1, 28, 28)
    model = ConvAutoEncoder(in_channels=1, latent_dim=64, base_channels=32)
    recon = model(x)
    z = model.encode(x)
    print(z.shape)
    print(recon.shape)

