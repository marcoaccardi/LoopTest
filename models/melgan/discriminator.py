# discriminator.py

import torch
import torch.nn as nn
from models.melgan.weights import WNConv1d, weights_init

class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf: int, n_layers: int, downsampling_factor: int):
        """
        N-layer discriminator with downsampling.
        Args:
            ndf (int): Number of discriminator filters in the first layer.
            n_layers (int): Number of layers in the discriminator.
            downsampling_factor (int): Factor by which the input is downsampled.
        """
        super().__init__()
        model = nn.ModuleDict()

        # First layer
        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass of the discriminator.
        Args:
            x (torch.Tensor): Input waveform.
        Returns:
            list: List of feature maps produced by the discriminator at each layer.
        """
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self, num_D: int, ndf: int, n_layers: int, downsampling_factor: int):
        """
        Multi-scale discriminator with multiple discriminators operating at different scales.
        Args:
            num_D (int): Number of discriminators (scales).
            ndf (int): Number of discriminator filters in the first layer.
            n_layers (int): Number of layers in each discriminator.
            downsampling_factor (int): Downsampling factor between discriminators.
        """
        super().__init__()
        self.model = nn.ModuleDict()

        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass of the multi-scale discriminator.
        Args:
            x (torch.Tensor): Input waveform.
        Returns:
            list: List of feature maps from all discriminators at each scale.
        """
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results
