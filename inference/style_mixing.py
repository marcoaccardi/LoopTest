import os
import torch
from tqdm import tqdm
from utils import load_mean_std, save_audio
from melgan_wrapper import load_vocoder

@torch.no_grad()
def style_mixing(args, generator, step, mean_style, n_source, n_target, device, j):
    index = 2
    os.makedirs(f'./generated_interpolation_one_bar_{index}/{j}', exist_ok=True)

    mean, std = load_mean_std(args.data_path, 80, device)
    vocoder = load_vocoder(device, './melgan/args.yml', './melgan/best_netG.pt')

    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)

    source_image, _ = generator([source_code], truncation=args.truncation, truncation_latent=mean_style)
    target_image, _ = generator([target_code], truncation=args.truncation, truncation_latent=mean_style)

    for i in range(n_source):
        de_norm = source_image[i] * std + mean
        audio_output = vocoder(de_norm)
        save_audio(f'./generated_interpolation_one_bar_{index}/{j}/source_{i}.wav', audio_output.squeeze().cpu().numpy(), 44100)

    for i in range(n_target):
        de_norm = target_image[i] * std + mean
        audio_output = vocoder(de_norm)
        save_audio(f'./generated_interpolation_one_bar_{index}/{j}/target_{i}.wav', audio_output.squeeze().cpu().numpy(), 44100)
