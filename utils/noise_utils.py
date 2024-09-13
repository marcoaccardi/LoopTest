# noise_utils.py

import torch
import random

def make_noise(batch: int, latent_dim: int, n_noise: int, device: torch.device) -> torch.Tensor:
    """
    Generates random noise vectors for input to the generator.
    
    Args:
        batch (int): Number of noise samples to generate (batch size).
        latent_dim (int): Dimensionality of the latent space (noise vector size).
        n_noise (int): Number of different noise vectors to generate.
        device (torch.device): The device on which to create the noise.
    
    Returns:
        torch.Tensor: Randomly generated noise vectors.
    """
    if n_noise == 1:
        # Generate a single noise vector
        return torch.randn(batch, latent_dim, device=device)

    # Generate multiple noise vectors and return them as a list
    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
    return noises


def mixing_noise(batch: int, latent_dim: int, prob: float, device: torch.device) -> list:
    """
    Generates noise with a probability of latent code mixing.
    
    Args:
        batch (int): Number of noise samples to generate (batch size).
        latent_dim (int): Dimensionality of the latent space (noise vector size).
        prob (float): Probability of mixing two different noise vectors.
        device (torch.device): The device on which to create the noise.
    
    Returns:
        list: A list of one or two noise vectors, depending on the mixing probability.
    """
    if prob > 0 and random.random() < prob:
        # Mix two noise vectors with a certain probability
        return make_noise(batch, latent_dim, 2, device)

    else:
        # Return a single noise vector if mixing doesn't occur
        return [make_noise(batch, latent_dim, 1, device)]
