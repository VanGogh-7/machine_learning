import math

import torch
from torch import nn
from torch.nn import functional as F


def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device) * -scale
        )
        embeddings = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=1)
        if self.embedding_dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return self.mlp(embeddings)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            make_group_norm(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            make_group_norm(out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.time_projection = nn.Linear(time_embedding_dim, out_channels)

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        features = self.downsample(inputs)
        features = self.conv(features)
        time_features = self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        return features + time_features


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)
        self.time_projection = nn.Linear(time_embedding_dim, out_channels)

    def forward(
        self,
        decoder_features: torch.Tensor,
        skip_features: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        decoder_features = self.upsample(decoder_features)
        height_diff = skip_features.size(2) - decoder_features.size(2)
        width_diff = skip_features.size(3) - decoder_features.size(3)
        if height_diff != 0 or width_diff != 0:
            decoder_features = F.pad(
                decoder_features,
                [
                    width_diff // 2,
                    width_diff - width_diff // 2,
                    height_diff // 2,
                    height_diff - height_diff // 2,
                ],
            )
        features = torch.cat([skip_features, decoder_features], dim=1)
        features = self.conv(features)
        time_features = self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        return features + time_features


class SimpleUNetNoisePredictor(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 64,
        time_embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        # The model receives a noisy image x_t and predicts the noise in it.
        self.initial = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        # The encoder captures context while preserving skip features.
        self.down1 = DownBlock(base_channels, base_channels * 2, time_embedding_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_embedding_dim)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 4)

        # The decoder combines denoising context with high-resolution skip details.
        self.up1 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            time_embedding_dim,
        )
        self.up2 = UpBlock(
            base_channels * 2,
            base_channels,
            base_channels,
            time_embedding_dim,
        )
        self.output = nn.Conv2d(base_channels, image_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)
        skip1 = self.initial(x)
        skip2 = self.down1(skip1, time_embedding)
        features = self.down2(skip2, time_embedding)
        features = self.bottleneck(features)
        features = self.up1(features, skip2, time_embedding)
        features = self.up2(features, skip1, time_embedding)
        # Predicting noise gives a simple MSE objective for diffusion training.
        return self.output(features)


if __name__ == "__main__":
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 1000, (4,))
    model = SimpleUNetNoisePredictor(
        image_channels=1,
        base_channels=64,
        time_embedding_dim=128,
    )
    pred_noise = model(x, t)
    print(pred_noise.shape)

