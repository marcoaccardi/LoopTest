# loss_functions.py

import torch
from torch import autograd
from torch.nn import functional as F

def d_logistic_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """
    Calculates the logistic loss for the discriminator in GAN training.
    
    Args:
        real_pred (torch.Tensor): Predictions of the discriminator on real images.
        fake_pred (torch.Tensor): Predictions of the discriminator on fake images.
    
    Returns:
        torch.Tensor: Logistic loss for the discriminator.
    """
    real_loss = F.softplus(-real_pred)  # Real images should result in low predictions
    fake_loss = F.softplus(fake_pred)   # Fake images should result in high predictions

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
    """
    R1 regularization loss for the discriminator, used to penalize the gradients 
    on real images (helps stabilize GAN training).
    
    Args:
        real_pred (torch.Tensor): Predictions of the discriminator on real images.
        real_img (torch.Tensor): The real images.
    
    Returns:
        torch.Tensor: The R1 regularization loss.
    """
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    """
    Non-saturating loss for the generator in GAN training.
    
    Args:
        fake_pred (torch.Tensor): Predictions of the discriminator on fake images.
    
    Returns:
        torch.Tensor: Non-saturating loss for the generator.
    """
    return F.softplus(-fake_pred).mean()
