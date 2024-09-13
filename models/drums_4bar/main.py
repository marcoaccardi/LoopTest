import torch
from generator import Generator
from discriminator import Discriminator

if __name__ == "__main__":
    # Initialize models
    generator = Generator(size=64, style_dim=512, n_mlp=8, channel_multiplier=2)
    discriminator = Discriminator(size=64)

    # Example inference
    noise = torch.randn(1, 2, 512)
    fake_img, _ = generator([noise])
    fake_output = discriminator(fake_img)

    # Use pdb for debugging if needed
    import pdb; pdb.set_trace()
