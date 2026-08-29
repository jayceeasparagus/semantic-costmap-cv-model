"""A compact U-Net encoder-decoder for semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class DoubleConv(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class EncoderBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.features = DoubleConv(input_channels, output_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.features(inputs)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=2, stride=2
        )
        self.features = DoubleConv(output_channels + skip_channels, output_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.upsample(inputs)
        if inputs.shape[-2:] != skip.shape[-2:]:
            inputs = functional.interpolate(
                inputs,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.features(torch.cat((skip, inputs), dim=1))


class SemanticSegmentationUNet(nn.Module):
    """U-Net that emits one logit map per semantic class."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        channels = [base_channels * factor for factor in (1, 2, 4, 8)]

        self.encoder1 = EncoderBlock(input_channels, channels[0])
        self.encoder2 = EncoderBlock(channels[0], channels[1])
        self.encoder3 = EncoderBlock(channels[1], channels[2])
        self.encoder4 = EncoderBlock(channels[2], channels[3])
        self.bottleneck = DoubleConv(channels[3], channels[3] * 2)
        self.decoder4 = DecoderBlock(channels[3] * 2, channels[3], channels[3])
        self.decoder3 = DecoderBlock(channels[3], channels[2], channels[2])
        self.decoder2 = DecoderBlock(channels[2], channels[1], channels[1])
        self.decoder1 = DecoderBlock(channels[1], channels[0], channels[0])
        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skip1, inputs = self.encoder1(inputs)
        skip2, inputs = self.encoder2(inputs)
        skip3, inputs = self.encoder3(inputs)
        skip4, inputs = self.encoder4(inputs)
        inputs = self.bottleneck(inputs)
        inputs = self.decoder4(inputs, skip4)
        inputs = self.decoder3(inputs, skip3)
        inputs = self.decoder2(inputs, skip2)
        inputs = self.decoder1(inputs, skip1)
        return self.classifier(inputs)
