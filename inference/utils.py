import numpy as np
import torch

def load_mean_std(data_path, feature_dim, device):
    """
    Loads mean and standard deviation for normalization from .npy files.

    Args:
        data_path (str): Path to the directory containing mean and std `.npy` files.
        feature_dim (int): Dimensionality of the features (e.g., number of mel bands).
        device (torch.device): Device to load the data onto ('cuda' or 'cpu').

    Returns:
        mean (torch.Tensor): Tensor containing the mean for normalization.
        std (torch.Tensor): Tensor containing the standard deviation for normalization.
    
    Example:
        mean, std = load_mean_std('./data', 80, 'cuda')
    """
    mean_fp = f'{data_path}/mean.mel.npy'
    std_fp = f'{data_path}/std.mel.npy'

    mean = torch.from_numpy(np.load(mean_fp)).float().view(1, feature_dim, 1).to(device)
    std = torch.from_numpy(np.load(std_fp)).float().view(1, feature_dim, 1).to(device)

    return mean, std

import soundfile as sf

def save_audio(file_path, audio_data, sample_rate=44100):
    """
    Saves a NumPy array containing audio data to a .wav file.

    Args:
        file_path (str): Path where the audio file should be saved.
        audio_data (numpy.ndarray): A NumPy array containing the audio signal to be saved.
        sample_rate (int, optional): The sample rate of the audio file. Defaults to 44100 Hz.

    Returns:
        None. The function writes the audio data to the specified file path.
    
    Example:
        save_audio('output.wav', audio_array, 44100)
    """
    sf.write(file_path, audio_data, sample_rate)
