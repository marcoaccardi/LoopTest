# utils.py

import torch
import torch.nn as nn


class NoiseInjection(nn.Module):
    """
    Injects random noise into the input image tensor.
    """
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        """
        Forward pass to inject noise into the image.

        Args:
            image (torch.Tensor): The input image tensor.
            noise (torch.Tensor): The noise tensor to inject. If None, it will generate random noise.

        Returns:
            torch.Tensor: Image with noise injected.
        """
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise

class ConstantInput(nn.Module):
    """
    Provides a constant input for the initial layer of the Generator.
    """
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        """
        Repeats the constant input for each batch element.

        Args:
            input (torch.Tensor): The input tensor to determine batch size.

        Returns:
            torch.Tensor: Repeated constant input for each batch.
        """
        batch = input.shape[0]
        return self.input.repeat(batch, 1, 1, 1)


