import torch
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        branch1_channels: int,
        branch2_reduce_channels: int,
        branch2_channels: int,
        branch3_reduce_channels: int,
        branch3_channels: int,
        branch4_channels: int,
    ) -> None:
        super().__init__()
        self.branch1 = ConvBNReLU(in_channels, branch1_channels, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, branch2_reduce_channels, kernel_size=1),
            ConvBNReLU(branch2_reduce_channels, branch2_channels, 3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, branch3_reduce_channels, kernel_size=1),
            ConvBNReLU(branch3_reduce_channels, branch3_channels, 5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, branch4_channels, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Parallel branches capture features at multiple receptive field sizes.
        branch_outputs = [
            self.branch1(inputs),
            self.branch2(inputs),
            self.branch3(inputs),
            self.branch4(inputs),
        ]
        # Concatenation preserves every branch's channels for the next layer.
        return torch.cat(branch_outputs, dim=1)


class InceptionCIFAR100(nn.Module):
    def __init__(self, num_classes: int = 100, dropout: float = 0.4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(3, 64, kernel_size=3, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, padding=1),
        )
        self.features = nn.Sequential(
            # 1x1 bottlenecks reduce computation before 3x3 and 5x5 convolutions.
            InceptionBlock(64, 32, 48, 64, 8, 16, 16),      # 128 channels
            InceptionBlock(128, 64, 64, 96, 16, 32, 32),    # 224 channels
            nn.MaxPool2d(kernel_size=2),                    # 32x32 -> 16x16
            InceptionBlock(224, 96, 64, 128, 16, 32, 32),   # 288 channels
            InceptionBlock(288, 128, 96, 160, 32, 64, 64),  # 416 channels
            nn.MaxPool2d(kernel_size=2),                    # 16x16 -> 8x8
            InceptionBlock(416, 160, 112, 192, 32, 64, 64), # 480 channels
            InceptionBlock(480, 192, 128, 256, 32, 96, 96), # 640 channels
        )
        self.classifier = nn.Sequential(
            # Adaptive average pooling avoids hard-coding the final spatial size.
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(640, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.features(features)
        return self.classifier(features)


if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 32)
    model = InceptionCIFAR100(num_classes=100)
    y = model(x)
    print(y.shape)
