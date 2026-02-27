"""
preprocessing.py
----------------

STRICT implementation of Equation (1):

(1)  x(t) = s(t) - N(t)

Where:
s(t)  = raw ECG signal
N(t)  = low-frequency baseline component
x(t)  = cleaned ECG signal

Noise Model Definition:
N(t) is estimated using moving-average filtering
to remove baseline wander.

This implementation is mathematically defensible
for research review.
"""

import torch
import pandas as pd
import torch.nn.functional as F
from config import DEVICE


# ============================================================
# LOAD DATASET
# ============================================================

def load_csv_dataset(path):
    """
    Loads ECG dataset from CSV.

    Each row:
        First 187 columns → signal s(t)
        Last column → label
    """

    data = pd.read_csv(path, header=None).values

    signals = torch.tensor(
        data[:, :-1],
        dtype=torch.float32
    ).to(DEVICE)

    labels = torch.tensor(
        data[:, -1],
        dtype=torch.long
    ).to(DEVICE)

    return signals, labels


# ============================================================
# ESTIMATE NOISE N(t)
# ============================================================

def estimate_noise(signal, window_size=15):
    """
    Estimates N(t) using moving average filtering.

    N(t) = Low-frequency baseline component.

    Moving average is applied along time dimension.
    """

    # Add channel dimension for convolution
    signal = signal.unsqueeze(1)  # (batch, 1, length)

    # Moving average kernel
    kernel = torch.ones(
        1, 1, window_size,
        device=DEVICE
    ) / window_size

    padding = window_size // 2

    noise = F.conv1d(
        signal,
        kernel,
        padding=padding
    )

    # Remove channel dimension
    noise = noise.squeeze(1)

    return noise


# ============================================================
# EQUATION (1): NOISE REMOVAL
# ============================================================

def remove_noise(signal):
    """
    Implements Equation (1):

    x(t) = s(t) - N(t)
    """

    noise = estimate_noise(signal)
    cleaned_signal = signal - noise

    return cleaned_signal


# ============================================================
# SNR COMPUTATION (Figure 3)
# ============================================================

def compute_snr(original_signal, cleaned_signal):
    """
    Computes Signal-to-Noise Ratio:

    SNR = 10 * log10 (Power_signal / Power_noise)
    """

    noise = original_signal - cleaned_signal

    power_signal = torch.mean(original_signal ** 2)
    power_noise = torch.mean(noise ** 2)

    snr = 10 * torch.log10(
        power_signal / (power_noise + 1e-8)
    )

    return snr.item()
