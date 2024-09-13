import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from tqdm import tqdm
from inference.generator_utils import load_generator, generate_sample
from melgan_wrapper import load_vocoder
from utils import load_mean_std, save_audio
from parsers import get_generate_audio_parser
import os
import numpy as np

def generate(args, g_ema, device, mean_latent):
    epoch = args.ckpt.split('.')[0]

    os.makedirs(f'{args.store_path}/{epoch}', exist_ok=True)
    os.makedirs(f'{args.store_path}/{epoch}/mel_80_320', exist_ok=True)

    mean, std = load_mean_std(args.data_path, 80, device)
    vocoder = load_vocoder(device, './melgan/args.yml', './melgan/best_netG.pt')

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)
            sample = generate_sample(g_ema, sample_z, args.truncation, mean_latent)
            np.save(f'{args.store_path}/{epoch}/mel_80_320/{i}.npy', sample.squeeze().cpu().numpy())

            de_norm = sample.squeeze(0) * std + mean
            audio_output = vocoder(de_norm)
            save_audio(f'{args.store_path}/{epoch}/{i}.wav', audio_output.squeeze().cpu().numpy(), 44100)
            print(f'Generated {i}th wav file')

if __name__ == "__main__":
    device = "cuda"
    parser = get_generate_audio_parser()
    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = load_generator(args.ckpt, args, device)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)

    if args.style_mixing:
        from style_mixing import style_mixing
        step = 0
        for j in range(20):
            style_mixing(args, g_ema, step, mean_latent, args.n_col, args.n_row, device, j)
