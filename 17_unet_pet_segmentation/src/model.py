import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, dropout)

    def forward(
        self,
        decoder_features: torch.Tensor,
        skip_features: torch.Tensor,
    ) -> torch.Tensor:
        decoder_features = self.up(decoder_features)
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
        # Skip connections concatenate high-resolution encoder details.
        features = torch.cat([skip_features, decoder_features], dim=1)
        return self.conv(features)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]
        # The encoder path captures increasingly broad image context.
        self.initial = DoubleConv(in_channels, channels[0], dropout)
        self.down1 = DownBlock(channels[0], channels[1], dropout)
        self.down2 = DownBlock(channels[1], channels[2], dropout)
        self.down3 = DownBlock(channels[2], channels[3], dropout)
        self.down4 = DownBlock(channels[3], channels[4], dropout)

        # The decoder path restores spatial resolution for pixel prediction.
        self.up1 = UpBlock(channels[4], channels[3], channels[3], dropout)
        self.up2 = UpBlock(channels[3], channels[2], channels[2], dropout)
        self.up3 = UpBlock(channels[2], channels[1], channels[1], dropout)
        self.up4 = UpBlock(channels[1], channels[0], channels[0], dropout)
        # A 1x1 convolution maps each pixel feature vector to class logits.
        self.output = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skip1 = self.initial(inputs)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        bottleneck = self.down4(skip4)

        features = self.up1(bottleneck, skip4)
        features = self.up2(features, skip3)
        features = self.up3(features, skip2)
        features = self.up4(features, skip1)
        return self.output(features)


if __name__ == "__main__":
    x = torch.randn(2, 3, 128, 128)
    model = UNet(in_channels=3, num_classes=3, base_channels=32)
    y = model(x)
    print(y.shape)
