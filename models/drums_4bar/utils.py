import torch

def make_kernel(k):
    """
    Create and normalize a 2D filter kernel from a 1D list.
    
    Args:
        k (list or tensor): 1D kernel values.
    
    Returns:
        torch.Tensor: Normalized 2D kernel.
    """
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


import torch
import torch.nn.functional as F

def upfirdn2d(x, kernel, up=1, down=1, pad=(0, 0)):
    """
    Upsample, apply FIR filter, and downsample 2D input.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch, channels, height, width).
        kernel (torch.Tensor): 2D filter kernel.
        up (int): Upsampling factor. Default is 1 (no upsampling).
        down (int): Downsampling factor. Default is 1 (no downsampling).
        pad (tuple): Padding for height and width (pad_height, pad_width). Default is (0, 0).
    
    Returns:
        torch.Tensor: Output tensor after upsample, FIR filtering, and downsample.
    """
    # Get padding values for height and width
    pad_height, pad_width = pad

    # Upsample
    if up > 1:
        x = F.interpolate(x, scale_factor=up, mode='nearest')

    # Pad input
    x = F.pad(x, (pad_width, pad_width, pad_height, pad_height), mode='constant')

    # Apply convolution with the kernel
    batch, channels, height, width = x.shape
    x = x.view(1, batch * channels, height, width)  # Group channels for depthwise conv
    kernel = kernel.view(1, 1, *kernel.shape)  # Shape kernel for 2D conv
    x = F.conv2d(x, kernel, groups=batch * channels)  # Depthwise convolution
    x = x.view(batch, channels, *x.shape[-2:])

    # Downsample
    if down > 1:
        x = x[:, :, ::down, ::down]

    return x
