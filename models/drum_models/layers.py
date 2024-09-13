import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

def make_kernel(k):
    """
    Create a 2D filter kernel from a 1D list and normalize it.

    Args:
        k (list or np.ndarray): 1D kernel values.

    Returns:
        torch.Tensor: Normalized 2D kernel tensor.
    """
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()
    return k

class PixelNorm(nn.Module):
    """
    Pixel-wise feature normalization layer.
    """
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class EqualConv2d(nn.Module):
    """
    2D convolution layer with equalized learning rate.

    Args:
        in_channel (int): Input channels.
        out_channel (int): Output channels.
        kernel_size (int): Size of convolution kernel.
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        return F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

class EqualLinear(nn.Module):
    """
    Fully connected layer with equalized learning rate and fused activation.
    """
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

