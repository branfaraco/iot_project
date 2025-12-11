import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2

    A basic building block for the U-Net architecture. It performs two
    convolution–batchnorm–ReLU operations in sequence. When building
    larger models, `DoubleConv` provides a concise way to specify the
    encoder and decoder submodules.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2D(nn.Module):
    """A 2D U-Net model for spatiotemporal traffic forecasting.

    Parameters
    ----------
    in_channels : int
        Number of input channels. For traffic forecasting, this is
        typically `history_steps * channels_per_frame`, e.g. 12 * 8.
    out_channels : int
        Number of output channels, equal to `future_steps`.
    base_ch : int, default 64
        Number of channels in the first convolutional layer. Deeper layers
        have multiples of this base channel count.
    """

    def __init__(self, in_channels: int, out_channels: int, base_ch: int = 64):
        super().__init__()

        self.down1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.bottleneck(self.pool3(x3))

        # decoder
        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)

        out = self.out_conv(x)
        return out