import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + residual)


class ResNetCIFAR10(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 10) -> None:
        super().__init__()
        self.current_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Residual stages preserve or downsample spatial dimensions.
        self.stage1 = self._make_stage(out_channels=64, n_blocks=2, stride=1)
        self.stage2 = self._make_stage(out_channels=128, n_blocks=2, stride=2)
        self.stage3 = self._make_stage(out_channels=256, n_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, n_classes)

    def _make_stage(
        self,
        out_channels: int,
        n_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        blocks = [BasicBlock(self.current_channels, out_channels, stride)]
        self.current_channels = out_channels
        blocks.extend(
            BasicBlock(out_channels, out_channels)
            for _ in range(n_blocks - 1)
        )
        return nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.stage1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.pool(outputs)
        outputs = torch.flatten(outputs, 1)
        return self.classifier(outputs)


if __name__ == "__main__":
    model = ResNetCIFAR10()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(y.shape)
