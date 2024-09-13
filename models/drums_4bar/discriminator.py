import math
import torch
from torch import nn
from layers import EqualConv2d, EqualLinear, ResBlock, ConvLayer  # Assuming these layers are defined in layers.py

class Discriminator(nn.Module):
    """
    Discriminator network for StyleGAN-based architectures.

    Args:
        size (int): Image size (e.g., 64, 128, 256).
        channel_multiplier (int): Multiplier for the number of channels at each layer.
        blur_kernel (list): Kernel to use for blurring in up/down-sampling layers.
    """
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        # Channel configurations based on image size
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # Initial convolution layer
        self.convs = [ConvLayer(1, channels[size], 1)]  # Assuming grayscale images (1 channel)

        log_size = int(math.log(size, 2))  # Log base 2 of image size
        in_channel = channels[size]

        # Add residual blocks for each layer from the size to 4x4
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        # Sequential block of convolution layers
        self.convs = nn.Sequential(*self.convs)

        # Minibatch standard deviation layer
        self.stddev_group = 4
        self.stddev_feat = 1

        # Final convolution and linear layers
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)  # Extra channel for minibatch stddev
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 5 * 5, channels[4], activation="fused_lrelu"),  # Linear layer with activation
            EqualLinear(channels[4], 1)  # Final output layer
        )

    def forward(self, input):
        """
        Forward pass of the Discriminator.

        Args:
            input (torch.Tensor): Input image tensor (batch, channels, height, width).

        Returns:
            torch.Tensor: Real or fake classification scores.
        """
        out = self.convs(input)

        # Minibatch standard deviation (optional layer for stabilizing GAN training)
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        # Final convolution and fully connected layers
        out = self.final_conv(out)
        out = out.view(batch, -1)  # Flatten the output
        out = self.final_linear(out)

        return out
