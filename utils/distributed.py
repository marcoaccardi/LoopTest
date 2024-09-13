import math
import pickle
import torch
from torch import distributed as dist
from torch.utils.data.sampler import Sampler

def get_rank() -> int:
    """
    Returns the rank of the current process in distributed training.
    
    Returns:
        int: Rank of the current process. If distributed training is not initialized, returns 0.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize() -> None:
    """
    Synchronizes all processes in distributed training. 
    Acts as a barrier to ensure all processes reach the same point before proceeding.
    
    Returns:
        None
    """
    if not dist.is_available() or not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    if world_size > 1:
        dist.barrier()


def get_world_size() -> int:
    """
    Returns the number of processes in the distributed training world.
    
    Returns:
        int: Number of processes. If distributed training is not initialized, returns 1.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduces the input tensor across all processes by summing the values. 
    This is used for gathering losses and other metrics across GPUs.
    
    Args:
        tensor (torch.Tensor): Tensor to be reduced.
    
    Returns:
        torch.Tensor: The reduced tensor.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def gather_grad(params):
    """
    Gathers gradients from all processes during distributed training, 
    averaging the gradients across all workers.
    
    Args:
        params: Model parameters that require gradient gathering.
    
    Returns:
        None
    """
    world_size = get_world_size()
    if world_size == 1:
        return

    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data.div_(world_size)


def all_gather(data):
    """
    Gathers data from all processes into a list on each process.
    
    Args:
        data: Data to be gathered.
    
    Returns:
        list: List of gathered data from all processes.
    """
    world_size = get_world_size()

    if world_size == 1:
        return [data]

    # Serialize data
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('cuda')

    local_size = torch.IntTensor([tensor.numel()]).to('cuda')
    size_list = [torch.IntTensor([0]).to('cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # Padding and gathering tensors
    tensor_list = [torch.ByteTensor(size=(max_size,)).to('cuda') for _ in size_list]
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to('cuda')
        tensor = torch.cat((tensor, padding), 0)

    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_loss_dict(loss_dict: dict) -> dict:
    """
    Reduces a dictionary of losses across all processes by summing them.
    
    Args:
        loss_dict (dict): Dictionary containing loss tensors to be reduced.
    
    Returns:
        dict: Dictionary of reduced losses.
    """
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []
        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses
