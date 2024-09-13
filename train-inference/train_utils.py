# train_utils.py

import os
import torch
from tqdm import tqdm
from utils.noise_utils import mixing_noise
from utils.model_utils import requires_grad, accumulate
from utils.loss_functions import d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize
from utils.distributed import reduce_loss_dict, get_rank, synchronize, get_world_size
from utils.non_leaking import augment, AdaptiveAugment
from torchvision import utils

def prepare_directories(args):
    """
    Create necessary directories for samples and checkpoints.
    
    Args:
        args: Command-line arguments including paths for saving directories.
    
    Returns:
        None
    """
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)


def initialize_training(args, generator, discriminator, g_ema, device):
    """
    Initialize variables and configurations needed for training, including EMA,
    data loaders, and adaptive augmentation.
    
    Args:
        args: Command-line arguments with training configuration.
        generator: The generator model.
        discriminator: The discriminator model.
        g_ema: Exponential moving average of the generator model.
        device: The device (GPU/CPU) to run the models on.
    
    Returns:
        Various initialized variables required during training.
    """
    # Initialize sample data and progress bar
    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    # Extract modules for distributed training
    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    # Initialize accumulators and augment probability
    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    # Setup adaptive augment if required
    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 256, device)

    # Generate random latent vectors for sampling
    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    return (pbar, mean_path_length, d_loss_val, r1_loss, g_loss_val, path_loss,
            path_lengths, mean_path_length_avg, loss_dict, g_module, d_module, 
            accum, ada_aug_p, r_t_stat, ada_augment, sample_z)


def save_checkpoint(generator, discriminator, g_ema, g_optim, d_optim, args, i, ada_aug_p):
    """
    Saves model checkpoints during training.
    
    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        g_ema: Exponential moving average of the generator.
        g_optim: Optimizer for the generator.
        d_optim: Optimizer for the discriminator.
        args: Command-line arguments containing checkpoint directory.
        i: Current iteration.
        ada_aug_p: Probability for adaptive augmentation.
    
    Returns:
        None
    """
    torch.save(
        {
            "g": generator.state_dict(),
            "d": discriminator.state_dict(),
            "g_ema": g_ema.state_dict(),
            "g_optim": g_optim.state_dict(),
            "d_optim": d_optim.state_dict(),
            "args": args,
            "ada_aug_p": ada_aug_p,
        },
        f"{args.checkpoint_dir}/{str(i).zfill(6)}.pt",
    )


def save_samples(g_ema, sample_z, args, i):
    """
    Save image samples generated by the model.
    
    Args:
        g_ema: Exponential moving average of the generator.
        sample_z: Latent vectors used for generating samples.
        args: Command-line arguments including sample directory.
        i: Current iteration.
    
    Returns:
        None
    """
    with torch.no_grad():
        g_ema.eval()
        sample, _ = g_ema([sample_z])
        utils.save_image(
            sample,
            f"{args.sample_dir}/{str(i).zfill(6)}.png",
            nrow=int(args.n_sample ** 0.5),
            normalize=True,
            range=(-1, 1),
        )


def adjust_ada_augment(args, real_pred, ada_augment, ada_aug_p, r_t_stat):
    """
    Adjusts the augmentation probability for adaptive augmentation during training.
    
    Args:
        args: Command-line arguments.
        real_pred: Predictions of the discriminator on real images.
        ada_augment: Adaptive augmentation module.
        ada_aug_p: Current augmentation probability.
        r_t_stat: Current augmentation statistics.
    
    Returns:
        Tuple of updated ada_aug_p and r_t_stat.
    """
    ada_aug_p = ada_augment.tune(real_pred)
    r_t_stat = ada_augment.r_t_stat

    return ada_aug_p, r_t_stat
