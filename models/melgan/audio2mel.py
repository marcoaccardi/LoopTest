# audio2mel.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        """
        Converts audio into Mel spectrograms.
        Args:
            n_fft (int): FFT window size.
            hop_length (int): Hop length for STFT.
            win_length (int): Window length for STFT.
            sampling_rate (int): Sampling rate of the audio.
            n_mel_channels (int): Number of mel filterbank channels.
            mel_fmin (float): Minimum mel frequency.
            mel_fmax (float): Maximum mel frequency.
        """
        super().__init__()
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()

        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Converts audio into log mel spectrograms.
        Args:
            audio (torch.Tensor): Input audio tensor.
        Returns:
            torch.Tensor: Log mel spectrogram tensor.
        """
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)

        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))

        return log_mel_spec
