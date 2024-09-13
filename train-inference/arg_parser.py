# arg_parser.py
import argparse

def get_args():
    """
    Parses command-line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    # Dataset and training configuration
    parser.add_argument("path", type=str, help="Path to the lmdb dataset")
    parser.add_argument("--iter", type=int, default=800000, help="Total training iterations")
    parser.add_argument("--batch", type=int, default=16, help="Batch sizes for each GPU")
    parser.add_argument("--n_sample", type=int, default=16, help="Number of samples generated during training")
    parser.add_argument("--size", type=int, default=256, help="Image sizes for the model")

    # Regularization and losses
    parser.add_argument("--r1", type=float, default=10, help="Weight of the R1 regularization")
    parser.add_argument("--path_regularize", type=float, default=2, help="Weight of the path length regularization")
    parser.add_argument("--path_batch_shrink", type=int, default=2, help="Batch size reducing factor for the path length regularization")
    parser.add_argument("--d_reg_every", type=int, default=16, help="Interval for applying R1 regularization")
    parser.add_argument("--g_reg_every", type=int, default=4, help="Interval for applying path length regularization")
    parser.add_argument("--mixing", type=float, default=0.9, help="Probability of latent code mixing")

    # Model and optimization
    parser.add_argument("--ckpt", type=str, default=None, help="Path to the checkpoints to resume training")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2, help="Channel multiplier factor for the model")

    # Logging and distribution
    parser.add_argument("--wandb", action="store_true", help="Use Weights and Biases logging")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--augment", action="store_true", help="Apply non-leaking augmentation")
    parser.add_argument("--augment_p", type=float, default=0, help="Probability of applying augmentation (0 = adaptive)")
    parser.add_argument("--ada_target", type=float, default=0.6, help="Target augmentation probability for adaptive augmentation")
    parser.add_argument("--ada_length", type=int, default=500 * 1000, help="Target duration to reach augmentation probability")
    parser.add_argument("--ada_every", type=int, default=256, help="Probability update interval for adaptive augmentation")

    # Directories
    parser.add_argument("--sample_dir", type=str, default='sample', help="Sample directory")
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoint', help="Checkpoint directory")

    return parser.parse_args()
