import math
from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torchaudio.transforms as AT

import numpy as np
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    return spec


def mel_scale(spec, fmax):
    global mel_basis
    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(spec.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


class Pipeline(nn.Module):
    def __init__(
        self,
        input_freq=22050,
        n_fft=1024,
        hop_length=256,
        n_mel=80,
        f_min=0,
        f_max=8000,
        spec_len=256,
        max_downsampling=5,
        augs={}
    ):
        super().__init__()

        self.spec = partial(
            spectrogram,
            n_fft=n_fft,
            num_mels=n_mel,
            sampling_rate=input_freq,
            hop_size=hop_length,
            win_size=hop_length * 4,
            fmin=f_min,
            fmax=f_max,
            center=False,
        )
        self.mel_scale = partial(mel_scale, fmax=8000)
        self.spec_len = spec_len
        self.max_downsampling = max_downsampling

    def forward(
        self, waveform: torch.Tensor, start_ptr=None, random_crop=True
    ) -> torch.Tensor:
        waveform = torch.clamp(waveform, -1.0, 1.0)

        spec = self.spec(waveform)
        mel = self.mel_scale(spec)

        # Crop
        *_, total_len = mel.shape
        if start_ptr is not None:
            start_ptr = start_ptr
        elif random_crop:
            start_ptr = np.random.randint(0, total_len - self.spec_len)
        else:
            center_ptr = total_len // 2
            start_ptr = center_ptr - self.spec_len // 2
        stop_ptr = min(start_ptr + self.spec_len, total_len)

        # Check length of spectrogram for downsampling
        factor = (stop_ptr - start_ptr) / (2**self.max_downsampling)
        if not factor.is_integer():
            scale = math.floor(factor)
            stop_ptr = scale * 2**self.max_downsampling + start_ptr

        mel = mel[:, :, start_ptr:stop_ptr]

        return mel