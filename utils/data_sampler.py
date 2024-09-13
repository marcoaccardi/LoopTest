# data_sampler.py

from torch.utils import data

def data_sampler(dataset, shuffle: bool, distributed: bool) -> data.Sampler:
    """
    Returns a data sampler based on whether the training is distributed or not, 
    and if the data should be shuffled.
    
    Args:
        dataset: Dataset to be sampled.
        shuffle (bool): If True, shuffle the data.
        distributed (bool): If True, use a distributed data sampler.

    Returns:
        data.Sampler: A data sampler (distributed, random, or sequential).
    """
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def sample_data(loader: data.DataLoader):
    """
    Yields batches of data indefinitely from the given data loader.
    
    Args:
        loader (data.DataLoader): DataLoader providing batches of data.
    
    Yields:
        Batches of data in an infinite loop.
    """
    while True:
        for batch in loader:
            yield batch
