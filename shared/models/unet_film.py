import torch
import torch.nn as nn

from .unet_base import DoubleConv  # reuse DoubleConv if needed


class WeatherFiLM(nn.Module):
    """Computes FiLM scaling and bias from a weather vector.

    Parameters
    ----------
    weather_dim : int
        Dimensionality of the conditioning vector (number of weather
        features).
    num_channels : int
        Number of channels in the feature map to be modulated.
    """

    def __init__(self, weather_dim: int, num_channels: int):
        super().__init__()
        self.gamma = nn.Linear(weather_dim, num_channels)
        self.beta = nn.Linear(weather_dim, num_channels)

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # w shape: (B, weather_dim)
        γ = self.gamma(w).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        β = self.beta(w).unsqueeze(-1).unsqueeze(-1)
        return γ, β


class DoubleConvFiLM(nn.Module):
    """(Conv → BN → ReLU) × 2 with optional FiLM modulation.

    If `weather_dim` is provided, the block applies FiLM after the two
    convolution layers using a `WeatherFiLM` module. Otherwise, it
    behaves like a standard `DoubleConv`.
    """

    def __init__(self, in_ch: int, out_ch: int, weather_dim: int | None = None):
        super().__init__()
        self.use_film = weather_dim is not None
        if self.use_film:
            self.film = WeatherFiLM(weather_dim, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, weather_vec: torch.Tensor | None = None) -> torch.Tensor:
        out = self.net(x)
        if self.use_film and weather_vec is not None:
            γ, β = self.film(weather_vec)
            out = γ * out + β
        return out


class UNet2D_FiLM(nn.Module):
    """A 2D U-Net model with FiLM conditioning.

    This class follows the same encoder–decoder structure as the base
    U-Net, but each convolution block uses `DoubleConvFiLM` so that
    weather vectors modulate the feature maps at every level of the
    network.

    Parameters
    ----------
    in_channels : int
        Number of input channels (including any LBCS channels and
        history steps).
    out_channels : int
        Number of output channels (equal to `future_steps`).
    weather_dim : int
        Dimensionality of the weather vector used for FiLM.
    base_ch : int, default 64
        Base number of convolution channels.
    """

    def __init__(self, in_channels: int, out_channels: int, weather_dim: int, base_ch: int = 64):
        super().__init__()
        self.down1 = DoubleConvFiLM(in_channels, base_ch, weather_dim)
        self.down2 = DoubleConvFiLM(base_ch, base_ch * 2, weather_dim)
        self.down3 = DoubleConvFiLM(base_ch * 2, base_ch * 4, weather_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConvFiLM(base_ch * 4, base_ch * 8, weather_dim)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.conv3 = DoubleConvFiLM(base_ch * 8, base_ch * 4, weather_dim)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConvFiLM(base_ch * 4, base_ch * 2, weather_dim)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.conv1 = DoubleConvFiLM(base_ch * 2, base_ch, weather_dim)
        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, weather_vec: torch.Tensor) -> torch.Tensor:
        # encoder with FiLM
        x1 = self.down1(x, weather_vec)
        x2 = self.down2(self.pool1(x1), weather_vec)
        x3 = self.down3(self.pool2(x2), weather_vec)
        x4 = self.bottleneck(self.pool3(x3), weather_vec)
        # decoder
        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x, weather_vec)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x, weather_vec)
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x, weather_vec)
        out = self.out_conv(x)
        return out
