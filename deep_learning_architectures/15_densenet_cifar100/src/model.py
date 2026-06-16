import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bottleneck_factor: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        bottleneck_channels = bottleneck_factor * growth_rate
        self.layers = nn.Sequential(
            # The bottleneck 1x1 convolution reduces computation before 3x3 conv.
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                bottleneck_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bottleneck_channels,
                growth_rate,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        new_features = self.dropout(self.layers(inputs))
        # Dense connections concatenate all previous features with new features.
        return torch.cat([inputs, new_features], dim=1)


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(current_channels, growth_rate, dropout=dropout))
            current_channels += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_channels = current_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TransitionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        compression: float,
    ) -> None:
        super().__init__()
        out_channels = int(in_channels * compression)
        if out_channels <= 0:
            raise ValueError("compression produced zero output channels.")
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            # Compression reduces channel count between dense blocks.
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DenseNetCIFAR100(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        growth_rate: int = 12,
        block_layers: tuple[int, ...] = (6, 12, 24, 16),
        compression: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0 < compression <= 1:
            raise ValueError("compression must be in the interval (0, 1].")

        initial_channels = 2 * growth_rate
        self.stem = nn.Conv2d(
            3,
            initial_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        channels = initial_channels
        stages = []
        for block_index, num_layers in enumerate(block_layers):
            dense_block = DenseBlock(channels, num_layers, growth_rate, dropout)
            stages.append(dense_block)
            channels = dense_block.out_channels
            if block_index != len(block_layers) - 1:
                transition = TransitionLayer(channels, compression)
                stages.append(transition)
                channels = transition.out_channels
        self.features = nn.Sequential(*stages)
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # Adaptive pooling avoids hard-coding the final spatial size.
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.features(features)
        return self.classifier(features)


if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 32)
    model = DenseNetCIFAR100(num_classes=100)
    y = model(x)
    print(y.shape)
