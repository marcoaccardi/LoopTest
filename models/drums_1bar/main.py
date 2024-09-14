from .generator import Generator
from discriminator import Discriminator
import torch
import os
os.environ['CUDA_HOME'] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3"


if __name__ == "__main__":
    generator = Generator(size=64, style_dim=512, n_mlp=8, channel_multiplier=2)
    noise = torch.randn(1, 16, 512).unbind(0)
    fake_img, _ = generator(noise)

    discriminator = Discriminator(size=64)
    fake_output = discriminator(fake_img)

    import pdb; pdb.set_trace()  # Use for debugging if necessary
