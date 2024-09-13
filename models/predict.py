import sys
import subprocess
import tempfile
from pathlib import Path
import os
import torch
import numpy as np
import soundfile as sf
import yaml
import cog  # COG framework for creating prediction APIs

from models.drums_4bar.main import Generator  # Import the drum model Generator
sys.path.append("./melgan")  # Append path for MelGAN

from melgan.modules import Generator  # Import MelGAN vocoder

# Utility function to read YAML configuration files
def read_yaml(fp):
    """
    Reads a YAML file and returns the loaded content.

    Args:
        fp (str): File path to the YAML file.
    
    Returns:
        dict: Loaded YAML content.
    """
    with open(fp) as file:
        return yaml.load(file, Loader=yaml.Loader)


class Predictor(cog.Predictor):
    """
    The Predictor class is a container for model inference using COG. It loads the 
    generator model and vocoder, generates audio from a random latent vector, and
    converts the output to a usable audio format (MP3).
    """

    def setup(self):
        """
        Sets up the model environment by loading the GAN generator (for generating 
        mel-spectrograms) and the vocoder (for converting the spectrograms to audio). 
        Loads normalization parameters to denormalize spectrograms before generating audio.
        """
        self.device = "cuda"  # Use GPU for inference
        checkpoint_path = "checkpoint-four-bar.pt"  # Path to the pre-trained GAN generator model checkpoint
        self.latent = 512  # Latent vector dimension for the GAN generator

        # Load the pre-trained GAN generator model
        self.g_ema = Generator(
            size=64,               # Output image size (related to the spectrogram size)
            style_dim=self.latent,  # Latent space dimension
            n_mlp=8,               # Number of layers in the MLP for style generation
            channel_multiplier=2,   # Multiplier for the number of channels
        ).to(self.device)

        # Load the generator's state dictionary from the checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.g_ema.load_state_dict(checkpoint["g_ema"])
        self.g_ema.eval()  # Set the model to evaluation mode

        # Load the normalization parameters (mean and standard deviation) for denormalizing mel-spectrograms
        data_path = "data/looperman_four_bar"
        feat_dim = 80  # Number of mel-spectrogram feature dimensions
        mean_fp = f"{data_path}/mean.mel.npy"
        std_fp = f"{data_path}/std.mel.npy"
        self.mean = (
            torch.from_numpy(np.load(mean_fp))
            .float()
            .view(1, feat_dim, 1)
            .to(self.device)
        )
        self.std = (
            torch.from_numpy(np.load(std_fp))
            .float()
            .view(1, feat_dim, 1)
            .to(self.device)
        )

        # Load and configure the MelGAN vocoder for converting spectrograms to audio
        vocoder_config_fp = "melgan/args.yml"
        vocoder_config = read_yaml(vocoder_config_fp)
        n_mel_channels = vocoder_config['n_mel_channels']
        ngf = vocoder_config['ngf']
        n_residual_layers = vocoder_config['n_residual_layers']
        self.sr = 44100  # Sample rate for the output audio

        # Initialize the vocoder and load its pre-trained weights
        self.vocoder = Generator_melgan(n_mel_channels, ngf, n_residual_layers).to(self.device)
        self.vocoder.eval()
        vocoder_param_fp = "melgan/best_netG.pt"
        self.vocoder.load_state_dict(torch.load(vocoder_param_fp))

    @cog.input("seed", type=int, default=-1, help="Random seed, -1 for random generation")
    def predict(self, seed):
        """
        Generates an audio sample based on a random seed. Uses the GAN generator to 
        create a mel-spectrogram, denormalizes it, and passes it through the vocoder 
        to produce audio. The output is saved in both WAV and MP3 formats.

        Args:
            seed (int): Random seed for generation (-1 for random seed).
        
        Returns:
            str: Path to the generated MP3 file.
        """
        # Set random seed for reproducibility
        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Prediction seed: {seed}")

        # Generate random latent vector and produce the mel-spectrogram
        sample_z = torch.randn(1, self.latent, device=self.device)
        sample, _ = self.g_ema([sample_z], truncation=1, truncation_latent=None)

        # Denormalize the generated spectrogram using preloaded mean and std
        de_norm = sample.squeeze(0) * self.std + self.mean

        # Convert the denormalized spectrogram to audio using the MelGAN vocoder
        audio_output = self.vocoder(de_norm)

        # Save the generated audio to a temporary directory
        out_dir = Path(tempfile.mkdtemp())
        wav_path = out_dir / "out.wav"
        mp3_path = out_dir / "out.mp3"

        try:
            # Save as WAV format
            sf.write(str(wav_path), audio_output.squeeze().detach().cpu().numpy(), self.sr)
            
            # Convert to MP3 using ffmpeg
            subprocess.check_output(
                ["ffmpeg", "-i", str(wav_path), str(mp3_path)],
            )
            return mp3_path  # Return the path to the generated MP3 file
        finally:
            wav_path.unlink(missing_ok=True)  # Clean up the temporary WAV file
