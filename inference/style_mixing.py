import os
import torch
from tqdm import tqdm
from utils import load_mean_std, save_audio
from melgan_wrapper import load_vocoder

@torch.no_grad()
def style_mixing(args, generator, step, mean_style, n_source, n_target, device, j):
    """
    Performs style mixing between source and target latent codes using a trained generator and vocoder.
    The function generates interpolated audio from a styleGAN generator, applies normalization, 
    converts the results to audio using a MelGAN vocoder, and saves the outputs.

    Args:
        args: argparse.Namespace
            Contains the necessary arguments such as truncation values, data path, etc.
        generator: torch.nn.Module
            Pre-trained StyleGAN generator model used to generate images/spectrograms.
        step: int
            Current training step (not directly used here but part of the architecture).
        mean_style: torch.Tensor
            Mean latent vector used for truncation during generation.
        n_source: int
            Number of source latent vectors to generate.
        n_target: int
            Number of target latent vectors to generate.
        device: torch.device
            The device on which to run the computations (e.g., 'cuda' or 'cpu').
        j: int
            Iteration or sample index for organizing saved files.

    Returns:
        None. The function saves generated audio files for source, target, and mixed images/spectrograms.
    
    Steps:
        1. Create directories for saving the generated audio files.
        2. Load mean and standard deviation for normalization from preprocessed data.
        3. Initialize the MelGAN vocoder for converting spectrograms to audio.
        4. Generate random latent codes for source and target images.
        5. Generate source and target spectrograms using the generator.
        6. Normalize the spectrograms and convert them to audio using the vocoder.
        7. Save the audio files for each source and target to the designated directories.
    """
    
    # Step 1: Create directory for saving generated audio
    index = 2
    os.makedirs(f'./generated_interpolation_one_bar_{index}/{j}', exist_ok=True)

    # Step 2: Load mean and standard deviation for normalization
    mean, std = load_mean_std(args.data_path, 80, device)
    
    # Step 3: Load MelGAN vocoder for audio synthesis from spectrograms
    vocoder = load_vocoder(device, './melgan/args.yml', './melgan/best_netG.pt')

    # Step 4: Generate random latent codes for source and target images
    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)

    # Step 5: Generate source and target images (spectrograms) using the generator
    source_image, _ = generator([source_code], truncation=args.truncation, truncation_latent=mean_style)
    target_image, _ = generator([target_code], truncation=args.truncation, truncation_latent=mean_style)

    # Step 6: Normalize source images and convert them to audio using the vocoder, then save
    for i in range(n_source):
        de_norm = source_image[i] * std + mean
        audio_output = vocoder(de_norm)
        save_audio(f'./generated_interpolation_one_bar_{index}/{j}/source_{i}.wav', audio_output.squeeze().cpu().numpy(), 44100)

    # Step 7: Normalize target images and convert them to audio using the vocoder, then save
    for i in range(n_target):
        de_norm = target_image[i] * std + mean
        audio_output = vocoder(de_norm)
        save_audio(f'./generated_interpolation_one_bar_{index}/{j}/target_{i}.wav', audio_output.squeeze().cpu().numpy(), 44100)
