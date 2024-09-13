# env_setup.py

import os
import torch
import torch.distributed as dist
from utils.distributed import synchronize

def setup_distributed_training(args):
    """
    Sets up the distributed training environment.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    n_gpu = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = n_gpu > 1

    if args.distributed:
        # Initialize distributed training
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()
