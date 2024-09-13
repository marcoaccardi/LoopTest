import torch
import os
import yaml
from models.melgan.generator import Generator_melgan

def read_yaml(fp):
    """Helper function to read a YAML file."""
    with open(fp) as file:
        return yaml.load(file, Loader=yaml.Loader)

def load_vocoder(device, config_path, vocoder_checkpoint):
    """Load the MelGAN vocoder with the given configuration and checkpoint."""
    vocoder_config = read_yaml(config_path)

    n_mel_channels = vocoder_config['n_mel_channels']
    ngf = vocoder_config['ngf']
    n_residual_layers = vocoder_config['n_residual_layers']

    vocoder = Generator_melgan(n_mel_channels, ngf, n_residual_layers).to(device)
    vocoder.load_state_dict(torch.load(vocoder_checkpoint))
    vocoder.eval()

    return vocoder
