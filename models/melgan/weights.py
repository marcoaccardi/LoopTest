# weights.py

import torch.nn as nn
from torch.nn.utils import weight_norm

def weights_init(m):
    """
    Initializes the weights of Conv and BatchNorm layers.
    Args:
        m: A layer of the model, typically a Conv or BatchNorm layer.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # Initialize Conv layers with normal distribution
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        # Initialize BatchNorm layers with normal distribution and fill biases with 0
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs) -> nn.Module:
    """
    Returns a 1D convolutional layer with weight normalization.
    Args:
        *args, **kwargs: Arguments for the Conv1D layer.
    Returns:
        nn.Module: Conv1D layer with weight normalization.
    """
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs) -> nn.Module:
    """
    Returns a 1D transposed convolutional layer with weight normalization.
    Args:
        *args, **kwargs: Arguments for the ConvTranspose1D layer.
    Returns:
        nn.Module: ConvTranspose1D layer with weight normalization.
    """
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))
