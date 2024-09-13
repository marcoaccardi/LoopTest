# generator.py

import torch
import torch.nn as nn
import numpy as np
from models.melgan.weights import WNConv1d, WNConvTranspose1d, weights_init
from models.melgan.resnet import ResnetBlock

class Generator_melgan(nn.Module):
    def __init__(self, input_size: int, ngf: int, n_residual_layers: int):
        """
        MelGAN Generator architecture.
        Args:
            input_size (int): Size of the input latent vector.
            ngf (int): Number of generator filters in the first layer.
            n_residual_layers (int): Number of residual layers per upsampling layer.
        """
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        # Initial block
        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            # Add residual layers
            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        # Final block
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MelGAN generator.
        Args:
            x (torch.Tensor): Input latent vector.
        Returns:
            torch.Tensor: Generated audio waveform.
        """
        return self.model(x)
