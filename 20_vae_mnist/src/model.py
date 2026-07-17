import torch
from torch import nn


class ConvVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 20,
        base_channels: int = 32,
        image_size: int = 28,
    ) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4.")

        self.encoded_size = image_size // 4
        self.encoded_channels = base_channels * 2
        self.feature_dim = self.encoded_channels * self.encoded_size * self.encoded_size

        # The encoder maps an image to the parameters of a latent distribution.
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
        self.mu_layer = nn.Linear(self.feature_dim, latent_dim)
        self.logvar_layer = nn.Linear(self.feature_dim, latent_dim)

        # The decoder maps sampled latent vectors back to image space.
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
            # MNIST tensors from ToTensor are in [0, 1], so sigmoid keeps
            # generated and reconstructed pixels in the same range.
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder_features(images)
        mu = self.mu_layer(features)
        logvar = self.logvar_layer(features)
        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        # The reparameterization trick allows gradients to flow through sampling.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.decoder_projection(z)
        features = features.view(
            z.size(0),
            self.encoded_channels,
            self.encoded_size,
            self.encoded_size,
        )
        return self.decoder_features(features)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(images)
        z = self.reparameterize(mu, logvar)
        reconstructed_images = self.decode(z)
        return reconstructed_images, mu, logvar


if __name__ == "__main__":
    x = torch.randn(4, 1, 28, 28)
    model = ConvVAE(in_channels=1, latent_dim=20, base_channels=32)
    recon, mu, logvar = model(x)
    z = model.reparameterize(mu, logvar)
    print(mu.shape)
    print(logvar.shape)
    print(z.shape)
    print(recon.shape)

