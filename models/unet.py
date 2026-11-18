"""Small U-Net for binary breast ultrasound lesion segmentation."""
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, base_channels: int = 16):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(1, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.bottleneck = ConvBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.out = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        z = self.bottleneck(self.pool(e3))
        z = self.dec3(torch.cat([self.up3(z), e3], dim=1))
        z = self.dec2(torch.cat([self.up2(z), e2], dim=1))
        z = self.dec1(torch.cat([self.up1(z), e1], dim=1))
        return self.out(z)
