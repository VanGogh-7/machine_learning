import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        image_dim: int = 784,
        hidden_dims: tuple[int, ...] = (256, 512, 1024),
        in_channels: int = 1,
        image_size: int = 28,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size

        layers: list[nn.Module] = []
        input_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, image_dim))
        # Tanh matches real images normalized to [-1, 1].
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Random noise vectors are transformed into fake image tensors.
        fake_images_flat = self.network(z)
        return fake_images_flat.view(z.size(0), self.in_channels, self.image_size, self.image_size)


class Discriminator(nn.Module):
    def __init__(
        self,
        image_dim: int = 784,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        input_dim = image_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        # Return raw logits. BCEWithLogitsLoss applies the sigmoid internally.
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.network(images)
        return logits.squeeze(1)


if __name__ == "__main__":
    z = torch.randn(4, 100)
    generator = Generator(latent_dim=100, image_dim=784)
    fake_images = generator(z)

    discriminator = Discriminator(image_dim=784)
    logits = discriminator(fake_images)

    print(fake_images.shape)
    print(logits.shape)

