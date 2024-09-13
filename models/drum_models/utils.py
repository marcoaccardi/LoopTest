# utils.py

import torch
import torch.nn as nn

def make_noise(log_size, device):
    """
    Generates random noise tensors of varying sizes for use in the generator.

    Args:
        log_size (int): The log size of the image dimensions (log2 of the image size).
        device (torch.device): The device to create the noise on.

    Returns:
        list: List of noise tensors.
    """
    noises = [torch.randn(1, 1, 4, 4, device=device)]  # Starting size (4x4)
    
    for i in range(3, log_size + 1):
        for _ in range(2):
            noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

    return noises

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
