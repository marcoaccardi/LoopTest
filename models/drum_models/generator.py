# generator.py

import math
import random  # <-- Import random module
import torch
from torch import nn
from .layers import PixelNorm, EqualLinear, StyledConv, ToRGB
from .utils import ConstantInput, NoiseInjection


class Generator(nn.Module):
    """
    StyleGAN2 Generator network that synthesizes images from latent vectors.

    Args:
        size (int): Output image size (e.g., 64, 128, etc.).
        style_dim (int): Dimensionality of the latent style vectors.
        n_mlp (int): Number of layers in the MLP for style generation.
        channel_multiplier (int): Channel multiplier for network scaling.
        blur_kernel (list): Kernel for blurring when upsampling.
        lr_mlp (float): Learning rate multiplier for the MLP layers.
    """
    def __init__(self, size, style_dim, n_mlp, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01):
        super().__init__()

        self.size = size
        self.style_dim = style_dim

        # Create the style MLP network
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"))
        self.style = nn.Sequential(*layers)

        # Channel configurations for each resolution
        self.channels = {
            4: 512, 8: 512, 16: 512, 32: 512,
            64: 256 * channel_multiplier, 128: 128 * channel_multiplier,
            256: 64 * channel_multiplier, 512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # Initial layers and ToRGB
        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1  # Number of layers

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        # Register noise buffers for each layer
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        # Convolutions and ToRGB layers for each resolution
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel)
            )
            self.convs.append(
                StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel)
            )
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def forward(self, styles, return_latents=False, inject_index=None, truncation=1, truncation_latent=None,
                input_is_latent=False, noise=None, randomize_noise=True):
        """
        Forward pass of the Generator.

        Args:
            styles (list of torch.Tensor): Latent style vectors.
            return_latents (bool): Whether to return latent codes.
            inject_index (int): Index to mix styles.
            truncation (float): Truncation trick for style vectors.
            truncation_latent (torch.Tensor): Latent vector for truncation.
            input_is_latent (bool): Whether the input is already in the latent space.
            noise (list): Noise inputs for each layer.
            randomize_noise (bool): Whether to randomize noise at each forward pass.

        Returns:
            torch.Tensor: Synthesized image.
            torch.Tensor or None: Latent codes (if return_latents is True).
        """
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

        if truncation < 1:
            styles = [(truncation_latent + truncation * (style - truncation_latent)) for style in styles]

        if len(styles) < 2:
            inject_index = self.n_latent
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
        else:
            inject_index = inject_index or random.randint(1, self.n_latent - 1)
            latent = torch.cat([styles[0].unsqueeze(1).repeat(1, inject_index, 1),
                                styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent
        return image, None
