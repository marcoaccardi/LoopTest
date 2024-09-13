import math
import torch
from torch import nn
from torch.nn import functional as F
from .utils import make_kernel, upfirdn2d
from op import fused_leaky_relu
class EqualConv2d(nn.Module):
    """
    2D convolution layer with equalized learning rate.
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


class EqualLinear(nn.Module):
    """
    Fully connected layer with equalized learning rate and optional activation.
    """
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.activation = activation

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        if self.activation:
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        return out


class NoiseInjection(nn.Module):
    """
    Injects noise into the input tensor with a learned weight.
    """
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            noise = torch.randn_like(image)
        return image + self.weight * noise

class ConstantInput(nn.Module):
    """
    Constant input for the initial layer of the Generator.
    
    Args:
        channel (int): Number of channels in the constant input tensor.
        size (int, optional): Size of the initial input. Default is 4x4.
    """
    def __init__(self, channel, size=4):
        super().__init__()
        # A learnable parameter that starts as a random tensor of shape (1, channel, size, size)
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        """
        Forward pass of ConstantInput.
        
        Args:
            input (torch.Tensor): The input tensor used to determine batch size.
        
        Returns:
            torch.Tensor: The constant input tensor repeated for each batch.
        """
        batch = input.shape[0]
        # Repeat the constant input for each batch element
        return self.input.repeat(batch, 1, 1, 1)

class ModulatedConv2d(nn.Module):
    """
    Modulated 2D convolution layer where weights are modulated by a style vector.
    
    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        style_dim (int): Dimensionality of the style vector used to modulate the convolution weights.
        demodulate (bool): Whether to apply demodulation to the convolution weights.
        upsample (bool): Whether to perform upsampling before the convolution.
        downsample (bool): Whether to perform downsampling after the convolution.
        blur_kernel (list): Kernel to use for blurring during up/downsampling.
    """
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        
        self.eps = 1e-8
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.downsample = downsample

        # Create the convolution weight parameter
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        
        # Equalized learning rate scaling
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        # Modulation is done via a fully connected layer that transforms the style vector
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        
        # Optional demodulation (StyleGAN trick)
        self.demodulate = demodulate

        # Blur for upsampling/downsampling
        if upsample:
            self.blur = Blur(blur_kernel, pad=(1, 1), upsample_factor=2)
        elif downsample:
            self.blur = Blur(blur_kernel, pad=(1, 1), upsample_factor=1)

    def forward(self, input, style):
        """
        Forward pass for ModulatedConv2d.
        
        Args:
            input (torch.Tensor): Input tensor.
            style (torch.Tensor): Style vector used for modulation.
        
        Returns:
            torch.Tensor: Output after modulated convolution.
        """
        batch, in_channel, height, width = input.shape

        # Modulate the weights based on the style
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.weight * self.scale * style

        # Demodulate the weights if enabled
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        # Reshape weight for batched convolution
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        # Perform convolution with optional upsampling/downsampling
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.kernel_size // 2, groups=batch)
        out = out.view(batch, self.out_channel, height, width)

        return out

class StyledConv(nn.Module):
    """
    Convolutional block with style modulation, noise injection, and activation.
    
    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        style_dim (int): Dimensionality of the style vector used to modulate the convolution weights.
        upsample (bool): Whether to upsample the input.
        blur_kernel (list): Kernel to use for blurring during upsampling.
        demodulate (bool): Whether to apply demodulation to the convolution weights.
    """
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True):
        super().__init__()

        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim, demodulate=demodulate, upsample=upsample, blur_kernel=blur_kernel)
        self.noise = NoiseInjection()
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise=None):
        """
        Forward pass for StyledConv.
        
        Args:
            input (torch.Tensor): Input tensor.
            style (torch.Tensor): Style vector used for modulation.
            noise (torch.Tensor, optional): Noise tensor for noise injection.
        
        Returns:
            torch.Tensor: Output after styled convolution.
        """
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    """
    Converts feature maps to RGB using a modulated 1x1 convolution.
    
    Args:
        in_channel (int): Number of input channels.
        style_dim (int): Dimensionality of the style vector used to modulate the convolution weights.
        upsample (bool): Whether to upsample the input.
        blur_kernel (list): Kernel to use for blurring during upsampling.
    """
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        
        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 1, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, input, style, skip=None):
        """
        Forward pass for ToRGB.
        
        Args:
            input (torch.Tensor): Input tensor.
            style (torch.Tensor): Style vector used for modulation.
            skip (torch.Tensor, optional): Output from the previous resolution level to be combined with the current output.
        
        Returns:
            torch.Tensor: Output in RGB space.
        """
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class Upsample(nn.Module):
    """
    Upsample the input using a custom blur kernel and a factor of 2.
    
    Args:
        kernel (list): 1D list or array to create a 2D blur kernel.
        factor (int): The upsampling factor, default is 2.
    """
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)  # Create the 2D kernel
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        """
        Forward pass of Upsample.
        
        Args:
            input (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Upsampled output tensor.
        """
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        return out

class ToRGB(nn.Module):
    """
    Converts feature maps to RGB using a modulated 1x1 convolution.
    
    Args:
        in_channel (int): Number of input channels.
        style_dim (int): Dimensionality of the style vector used to modulate the convolution weights.
        upsample (bool): Whether to upsample the input.
        blur_kernel (list): Kernel to use for blurring during upsampling.
    """
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)
            self.blur = Blur(blur_kernel, pad=(1, 1), upsample_factor=2)  # Use Blur here

        self.conv = ModulatedConv2d(in_channel, 1, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            skip = self.blur(skip)  # Apply blur after upsampling
            out = out + skip

        return out


class Blur(nn.Module):
    """
    Apply blur to the input using a custom blur kernel.
    
    Args:
        kernel (list): 1D list or array to create a 2D blur kernel.
        pad (tuple): Padding applied before applying the blur.
        upsample_factor (int): Factor by which the image was upsampled. Default is 1 (no upsampling).
    """
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)  # Convert 1D kernel into 2D kernel
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)  # Adjust kernel if upsampling
        self.register_buffer("kernel", kernel)  # Register kernel as a buffer (not a learnable parameter)
        self.pad = pad  # Set padding

    def forward(self, input):
        """
        Forward pass of Blur.
        
        Args:
            input (torch.Tensor): Input tensor to which blur is applied.
        
        Returns:
            torch.Tensor: Blurred output tensor.
        """
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out
