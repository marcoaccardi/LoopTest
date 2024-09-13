# resnet.py

import torch
import torch.nn as nn
from models.melgan.weights import WNConv1d

class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        """
        ResNet block with weight-normalized convolutions.
        Args:
            dim (int): Number of input/output channels.
            dilation (int): Dilation factor for convolution.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet block.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after ResNet block.
        """
        return self.shortcut(x) + self.block(x)
