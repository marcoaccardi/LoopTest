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

class ToRGB(nn.Module):
    """
    Converts the processed feature map to the final output image.

    Args:
        in_channel (int): Number of input channels.
        style_dim (int): Dimensionality of the style vector.
        upsample (bool, optional): Whether to upsample the feature map before conversion. Default is True.
        blur_kernel (list, optional): Kernel used for blurring during upsampling. Default is [1, 3, 3, 1].

    This class performs:
    - Modulated convolution to convert the feature map to a 1-channel output.
    - Optional upsampling to increase the resolution of the feature map before conversion.
    - Bias addition after the convolution.
    """
    
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        # Optional upsampling layer
        if upsample:
            self.upsample = Upsample(blur_kernel)

        # Modulated convolution to map the feature map to a 1-channel output
        self.conv = ModulatedConv2d(in_channel, 1, 1, style_dim, demodulate=False)

        # Bias added after convolution
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, input, style, skip=None):
        """
        Forward pass of ToRGB.
        
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channel, height, width).
            style (torch.Tensor): Style tensor that modulates the convolution weights.
            skip (torch.Tensor, optional): Optional skip connection for progressive growing.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, height, width).
        """
        # Apply modulated convolution to input
        out = self.conv(input, style)
        
        # Add bias to the result
        out = out + self.bias

        # If skip connection is provided, upsample and add it to the output
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        
        return out

class StyledConv(nn.Module):
    """
    A convolutional layer with modulation, noise injection, and activation.
    
    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        style_dim (int): Dimensionality of the style vector.
        upsample (bool, optional): Whether to upsample the input before convolution. Default is False.
        blur_kernel (list, optional): Kernel used for blurring during upsampling. Default is [1, 3, 3, 1].
        demodulate (bool, optional): Whether to apply demodulation after style modulation. Default is True.
    
    This class integrates:
    - Modulated convolution (`ModulatedConv2d`) which uses a style vector to modify the convolution kernel.
    - Noise injection to add noise at each layer, improving stochasticity.
    - Activation using fused LeakyReLU for non-linearity.
    """
    
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True):
        super().__init__()

        # Modulated convolution layer with style modulation
        self.conv = ModulatedConv2d(
            in_channel, out_channel, kernel_size, style_dim, 
            upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate
        )

        # Injecting noise after convolution
        self.noise = NoiseInjection()

        # Using Fused LeakyReLU for activation
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        """
        Forward pass of StyledConv.
        
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channel, height, width).
            style (torch.Tensor): Style tensor that modulates the convolution weights.
            noise (torch.Tensor, optional): Optional noise tensor. If not provided, random noise is generated.

        Returns:
            torch.Tensor: Output tensor after convolution, noise injection, and activation.
        """
        # Perform modulated convolution
        out = self.conv(input, style)
        
        # Inject noise into the output
        out = self.noise(out, noise=noise)
        
        # Apply activation function
        out = self.activate(out)
        
        return out

class Upsample(nn.Module):
    """
    Upsample the input using a custom blur kernel and factor of 2.

    Args:
        kernel (list): 1D list of kernel values.
        factor (int): Upsampling factor. Default is 2.
    """
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        self.kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', self.kernel)
        pad = kernel.shape[0] - factor
        self.pad = (pad + 1) // 2 + factor - 1, pad // 2

    def forward(self, input):
        return upfirdn2d(input, self.kernel, up=self.factor, pad=self.pad)
class ModulatedConv2d(nn.Module):
    """
    Modulated convolution layer with optional upsampling or downsampling.
    The weights of the convolution are modulated by a style vector.
    """
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, 
                 demodulate=True, upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.eps = 1e-8
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.downsample = downsample
        self.blur_kernel = blur_kernel

        # Convolution weights
        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        # Modulation: Learnable transformation of the style
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        # Demodulation flag
        self.demodulate = demodulate

        # Optionally add blur layers for upsampling/downsampling
        if upsample:
            self.blur = Blur(blur_kernel, pad=self.compute_padding())
        if downsample:
            self.blur = Blur(blur_kernel, pad=self.compute_padding())

    def compute_padding(self):
        # Compute padding for blur
        p = (len(self.blur_kernel) - self.upsample - self.kernel_size) + (self.kernel_size - 1)
        return (p + 1) // 2, p // 2

    def forward(self, input, style):
        # Get the batch size from the input
        batch, in_channel, height, width = input.shape

        # Modulate the convolution weights using the style vector
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.weight * style

        # Optional demodulation (StyleGAN trick)
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        # Reshape weight for batched convolution
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        # Apply convolution (with upsampling if needed)
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(input, weight, stride=2, padding=0, groups=batch)
            out = out.view(batch, self.out_channel, out.shape[2], out.shape[3])
            out = self.blur(out) if self.blur else out
        elif self.downsample:
            input = self.blur(input) if self.blur else input
            input = input.view(1, batch * in_channel, input.shape[2], input.shape[3])
            out = F.conv2d(input, weight, stride=2, padding=0, groups=batch)
            out = out.view(batch, self.out_channel, out.shape[2], out.shape[3])
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.kernel_size // 2, groups=batch)
            out = out.view(batch, self.out_channel, out.shape[2], out.shape[3])

        return out

import math
import torch
from torch import nn
from torch.nn import functional as F

class EqualConv2d(nn.Module):
    """
    A convolutional layer with equalized learning rate.
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None

    def forward(self, input):
        return F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

# Add other layers like ModulatedConv2d, NoiseInjection, etc.
