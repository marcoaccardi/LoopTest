# train.py

import os
import torch
from tqdm import tqdm
from torchvision import transforms, utils
import numpy as np

# Import utility functions
from utils.data_sampler import sample_data
from utils.distributed import reduce_sum
from utils.noise_utils import mixing_noise
from utils.model_utils import requires_grad, accumulate
from utils.loss_functions import d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize
from utils.distributed import reduce_loss_dict, get_rank, synchronize, get_world_size
from utils.non_leaking import augment, AdaptiveAugment
from .train_utils import (prepare_directories, initialize_training, 
                               save_checkpoint, save_samples, adjust_ada_augment)

# Import models and dataset
from models.drums_1bar.main import Generator, Discriminator
from utils.dataset import MultiResolutionDataset_drum

# Import argument parser and environment setup
from .arg_parser import get_args     # <-- Import get_args from arg_parser.py
from .env_setup import setup_distributed_training 

# Handle Weights and Biases (wandb) for logging
try:
    import wandb
except ImportError:
    wandb = None


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    """
    Main training loop for StyleGAN2, incorporating generator and discriminator training, 
    as well as regularization and augmentation adjustments.

    Args:
        args: Training configuration and command-line arguments.
        loader: DataLoader for the dataset.
        generator: The generator model.
        discriminator: The discriminator model.
        g_optim: Optimizer for the generator.
        d_optim: Optimizer for the discriminator.
        g_ema: Exponential moving average of the generator.
        device: The device (GPU/CPU) to run the models on.

    Returns:
        None
    """
    prepare_directories(args)
    loader = sample_data(loader)

    (pbar, mean_path_length, d_loss_val, r1_loss, g_loss_val, path_loss, path_lengths, 
     mean_path_length_avg, loss_dict, g_module, d_module, accum, ada_aug_p, r_t_stat, 
     ada_augment, sample_z) = initialize_training(args, generator, discriminator, g_ema, device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Training completed!")
            break

        # Get real images and apply preprocessing (normalization)
        real_img = next(loader).to(device)
        mean_fp = os.path.join(args.path, 'mean.mel.npy')
        std_fp = os.path.join(args.path, 'std.mel.npy')
        feat_dim = 80
        mean = torch.from_numpy(np.load(mean_fp)).float().to(device).view(1, feat_dim, 1)
        std = torch.from_numpy(np.load(std_fp)).float().to(device).view(1, feat_dim, 1)
        real_img = (real_img - mean) / std

        # Train the discriminator
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)
        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        # Adjust adaptive augmentation probability if required
        if args.augment and args.augment_p == 0:
            ada_aug_p, r_t_stat = adjust_ada_augment(args, real_pred, ada_augment, ada_aug_p, r_t_stat)

        # Discriminator regularization (R1 loss)
        if i % args.d_reg_every == 0:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()

        loss_dict["r1"] = r1_loss

        # Train the generator
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # Generator regularization (Path length regularization)
        if i % args.g_reg_every == 0:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss
            weighted_path_loss.backward()
            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        # Update exponential moving average of generator
        accumulate(g_ema, g_module, accum)

        # Reduce losses across GPUs (if distributed)
        loss_reduced = reduce_loss_dict(loss_dict)

        # Logging (can be adjusted for WandB or other frameworks)
        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {loss_reduced['d'].mean().item():.4f}; g: {loss_reduced['g'].mean().item():.4f}; "
                    f"r1: {loss_reduced['r1'].mean().item():.4f}; path: {loss_reduced['path'].mean().item():.4f}; "
                    f"mean path: {mean_path_length_avg:.4f}; augment: {ada_aug_p:.4f}"
                )
            )

            if i % 100 == 0:
                save_samples(g_ema, sample_z, args, i)

            if i % 10000 == 0:
                save_checkpoint(g_module, d_module, g_ema, g_optim, d_optim, args, i, ada_aug_p)


if __name__ == "__main__":
    # Parse command-line arguments
    args = get_args()

    # Setup distributed training if applicable
    setup_distributed_training(args)

    # Set device to CUDA
    device = "cuda"

    # Initialize models
    generator = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    # Optimizers
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    # Load checkpoint if specified
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    # Data loader
    transform = transforms.Compose([
        # Example transform setup
        transforms.ToTensor(),
    ])
    dataset = MultiResolutionDataset_drum(args.path, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sample_data(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    # Initialize Weights and Biases (if applicable)
    if get_rank() == 0 and args.wandb:
        wandb.init(project="stylegan2")

    # Start training
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
