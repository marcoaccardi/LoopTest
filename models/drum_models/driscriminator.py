# discriminator.py
import math
import torch
from torch import nn
from .layers import ConvLayer, ResBlock, EqualLinear

class Discriminator(nn.Module):
    """
    The main Discriminator network to classify real and fake images.
    
    Args:
        size (int): Image size (e.g., 64, 128, 256).
        channel_multiplier (int): Channel multiplier for various layers.
    """
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512, 8: 512, 16: 512, 32: 512,
            64: 256 * channel_multiplier, 128: 128 * channel_multiplier,
            256: 64 * channel_multiplier, 512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(1, channels[size], 1)]
        log_size = int(math.log(size, 2))

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 5 * 20, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)
        batch, channel, height, width = out.shape

        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8).mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        
        out = self.final_conv(out).view(batch, -1)
        return self.final_linear(out)
