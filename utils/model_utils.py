# model_utils.py
import torch
import torch.nn as nn
def requires_grad(model: nn.Module, flag: bool = True):
    """
    Set the `requires_grad` flag for all parameters of the given model.
    
    Args:
        model (nn.Module): The model whose parameters' gradients are to be modified.
        flag (bool): If True, enable gradients; if False, disable gradients.
    
    Returns:
        None
    """
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1: nn.Module, model2: nn.Module, decay: float = 0.999):
    """
    Accumulates parameters from `model2` into `model1` with exponential decay.
    Used to maintain a moving average of model weights (EMA).
    
    Args:
        model1 (nn.Module): Model where the accumulated parameters will be stored.
        model2 (nn.Module): Model from which parameters will be accumulated.
        decay (float): Decay rate for exponential moving average.
    
    Returns:
        None
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        # Apply exponential decay to accumulate parameters
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
