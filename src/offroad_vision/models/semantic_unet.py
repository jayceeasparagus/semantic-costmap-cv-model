import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class SemanticUNet(nn.Module):
    def __init__(self, num_classes=5, base_channels=32):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.encoder1 = ConvBlock(
            3,
            base_channels,
        )

        self.encoder2 = ConvBlock(
            base_channels,
            base_channels * 2,
        )

        self.encoder3 = ConvBlock(
            base_channels * 2,
            base_channels * 4,
        )

        self.encoder4 = ConvBlock(
            base_channels * 4,
            base_channels * 8,
        )

        self.bottleneck = ConvBlock(
            base_channels * 8,
            base_channels * 16,
        )

        self.up4 = nn.ConvTranspose2d(
            base_channels * 16,
            base_channels * 8,
            kernel_size=2,
            stride=2,
            bias=False,
        )

        self.decoder4 = ConvBlock(
            base_channels * 16,
            base_channels * 8,
        )

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2,
            bias=False,
        )

        self.decoder3 = ConvBlock(
            base_channels * 8,
            base_channels * 4,
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
            bias=False,
        )

        self.decoder2 = ConvBlock(
            base_channels * 4,
            base_channels * 2,
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )

        self.decoder1 = ConvBlock(
            base_channels * 2,
            base_channels,
        )

        self.classifier = nn.Conv2d(
            base_channels,
            num_classes,
            kernel_size=1,
        )

    def forward(self, inputs):
        encoder1 = self.encoder1(inputs)

        encoder2 = self.encoder2(
            self.pool(encoder1)
        )

        encoder3 = self.encoder3(
            self.pool(encoder2)
        )

        encoder4 = self.encoder4(
            self.pool(encoder3)
        )

        bottleneck = self.bottleneck(
            self.pool(encoder4)
        )

        decoder4 = self.up4(bottleneck)
        decoder4 = torch.cat(
            [decoder4, encoder4],
            dim=1,
        )
        decoder4 = self.decoder4(decoder4)

        decoder3 = self.up3(decoder4)
        decoder3 = torch.cat(
            [decoder3, encoder3],
            dim=1,
        )
        decoder3 = self.decoder3(decoder3)

        decoder2 = self.up2(decoder3)
        decoder2 = torch.cat(
            [decoder2, encoder2],
            dim=1,
        )
        decoder2 = self.decoder2(decoder2)

        decoder1 = self.up1(decoder2)
        decoder1 = torch.cat(
            [decoder1, encoder1],
            dim=1,
        )
        decoder1 = self.decoder1(decoder1)

        return self.classifier(decoder1)