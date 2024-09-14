import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
os.environ['CUDA_HOME'] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3"
from models.drums_1bar.main import Generator

def load_generator(ckpt, args, device):
    """Load the generator model from checkpoint."""
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(ckpt)
    g_ema.load_state_dict(checkpoint["g_ema"])
    return g_ema

def generate_sample(g_ema, sample_z, truncation, mean_latent):
    """Generate a sample using the generator."""
    sample, _ = g_ema([sample_z], truncation=truncation, truncation_latent=mean_latent)
    return sample
