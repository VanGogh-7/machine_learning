import torch
from torch import nn


def make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            # Standard convolution mixes spatial and channel information together.
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise convolution applies one spatial filter per input channel.
            ConvBNReLU(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels,
            ),
            # Pointwise convolution mixes information across channels cheaply.
            ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class MobileNetV1(nn.Module):
    def __init__(
        self,
        num_classes: int = 102,
        width_multiplier: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if width_multiplier <= 0:
            raise ValueError("width_multiplier must be positive.")

        def channels(base_channels: int) -> int:
            return make_divisible(base_channels * width_multiplier)

        channel_plan = [
            (32, 2),
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (1024, 2),
            (1024, 1),
        ]
        first_channels = channels(channel_plan[0][0])
        layers: list[nn.Module] = [
            # Width multiplier scales channel counts for efficiency experiments.
            ConvBNReLU(3, first_channels, kernel_size=3, stride=2)
        ]
        in_channels = first_channels
        for base_channels, stride in channel_plan[1:]:
            out_channels = channels(base_channels)
            # Depthwise separable conv reduces computation versus standard conv.
            layers.append(DepthwiseSeparableConv(in_channels, out_channels, stride))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            # Adaptive average pooling avoids depending on a fixed feature size.
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


if __name__ == "__main__":
    x = torch.randn(4, 3, 128, 128)
    model = MobileNetV1(num_classes=102)
    y = model(x)
    print(y.shape)
