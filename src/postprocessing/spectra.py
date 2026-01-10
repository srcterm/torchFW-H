"""
Spectral analysis utilities for acoustic signals.

Provides functions to compute:
- SPL (Sound Pressure Level) spectra using Welch's method
- OASPL (Overall Sound Pressure Level) from time-domain RMS
- Band-limited OASPL for specific frequency ranges

All functions operate on PyTorch tensors and support batch processing.
"""

from typing import Tuple
import torch
from torch import Tensor

# Ref. pressure for air (20 μPa)
P_REF_DEFAULT = 20e-6


def _hann_window(size: int, device: torch.device) -> Tensor:
    n = torch.arange(size, device=device, dtype=torch.float32)
    return 0.5 * (1 - torch.cos(2 * torch.pi * n / (size - 1)))

def compute_spl(
    signal: Tensor,
    dt: float,
    p_ref: float = P_REF_DEFAULT,
    nperseg: int = 1024,
    overlap: float = 0.5
) -> Tuple[Tensor, Tensor]:
    """
    Compute Sound Pressure Level spectrum using Welch's method.
    Uses overlapping segments with Hann windowing and averaging for
    reduced variance spectral estimation.
    """
    device = signal.device
    dtype = signal.dtype

    *batch_dims, N_t = signal.shape

    if nperseg > N_t:
        nperseg = N_t
    step = int(nperseg * (1 - overlap))
    step = max(1, step)
    n_segments = (N_t - nperseg) // step + 1
    if n_segments < 1:
        n_segments = 1
        step = 0
    win = _hann_window(nperseg, device)
    win_sq_sum = (win ** 2).sum()

    fs = 1.0 / dt
    freqs = torch.fft.rfftfreq(nperseg, dt, device=device)
    N_freq = freqs.shape[0]
    signal_flat = signal.reshape(-1, N_t)
    B = signal_flat.shape[0]
    psd_accum = torch.zeros(B, N_freq, device=device, dtype=dtype)

    for seg_idx in range(n_segments):
        start = seg_idx * step
        end = start + nperseg
        segment = signal_flat[:, start:end]
        segment_windowed = segment * win
        spectrum = torch.fft.rfft(segment_windowed, dim=-1)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        psd_accum = psd_accum + power

    psd_accum = psd_accum / n_segments
    # PSD normalize
    scale = fs * win_sq_sum
    psd = psd_accum / scale

    # One-sided scaling (double except DC and Nyquist)
    psd[:, 1:-1] = psd[:, 1:-1] * 2
    if nperseg % 2 != 0:
        psd[:, -1] = psd[:, -1] * 2

    # Convert to SPL
    psd = torch.clamp(psd, min=1e-40)
    spl = 10 * torch.log10(psd / (p_ref ** 2))
    spl = spl.reshape(*batch_dims, N_freq)
    return freqs, spl


def compute_oaspl(
    signal: Tensor,
    p_ref: float = P_REF_DEFAULT,
    axis: int = -1
) -> Tensor:
    """
    Compute Overall Sound Pressure Level from time-domain RMS.
    """
    p_sq_mean = (signal ** 2).mean(dim=axis)
    p_rms = torch.sqrt(p_sq_mean)
    p_rms = torch.clamp(p_rms, min=1e-40)
    return 20 * torch.log10(p_rms / p_ref)


def compute_oaspl_band(
    signal: Tensor,
    dt: float,
    f_low: float,
    f_high: float,
    p_ref: float = P_REF_DEFAULT
) -> Tensor:
    """
    Compute band-limited Overall Sound Pressure Level.
    Filters the signal to [f_low, f_high] Hz before computing OASPL.
    """
    device = signal.device
    *batch_dims, N_t = signal.shape
    spectrum = torch.fft.rfft(signal, dim=-1)
    freqs = torch.fft.rfftfreq(N_t, dt, device=device)

    mask = ((freqs >= f_low) & (freqs <= f_high)).float()
    for _ in batch_dims:
        mask = mask.unsqueeze(0)

    spectrum_filtered = spectrum * mask
    signal_filtered = torch.fft.irfft(spectrum_filtered, n=N_t, dim=-1)
    return compute_oaspl(signal_filtered, p_ref, axis=-1)


def compute_psd(
    signal: Tensor,
    dt: float,
    nperseg: int = 1024,
    overlap: float = 0.5
) -> Tuple[Tensor, Tensor]:
    """
    Compute Power Spectral Density using Welch's method.
    """
    device = signal.device
    dtype = signal.dtype
    *batch_dims, N_t = signal.shape

    if nperseg > N_t:
        nperseg = N_t
    step = int(nperseg * (1 - overlap))
    step = max(1, step)
    n_segments = (N_t - nperseg) // step + 1
    if n_segments < 1:
        n_segments = 1
        step = 0
    win = _hann_window(nperseg, device)
    win_sq_sum = (win ** 2).sum()

    fs = 1.0 / dt
    freqs = torch.fft.rfftfreq(nperseg, dt, device=device)
    N_freq = freqs.shape[0]
    signal_flat = signal.reshape(-1, N_t)
    B = signal_flat.shape[0]
    psd_accum = torch.zeros(B, N_freq, device=device, dtype=dtype)

    for seg_idx in range(n_segments):
        start = seg_idx * step
        end = start + nperseg
        segment = signal_flat[:, start:end]
        segment_windowed = segment * win
        spectrum = torch.fft.rfft(segment_windowed, dim=-1)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        psd_accum = psd_accum + power
    psd_accum = psd_accum / n_segments
    scale = fs * win_sq_sum
    psd = psd_accum / scale

    psd[:, 1:-1] = psd[:, 1:-1] * 2
    if nperseg % 2 != 0:
        psd[:, -1] = psd[:, -1] * 2

    psd = psd.reshape(*batch_dims, N_freq)
    return freqs, psd


def spl_to_psd(spl: Tensor, p_ref: float = P_REF_DEFAULT) -> Tensor:
    """Convert SPL (dB) to PSD (Pa2/Hz)."""
    return (p_ref ** 2) * (10 ** (spl / 10))


def psd_to_spl(psd: Tensor, p_ref: float = P_REF_DEFAULT) -> Tensor:
    """Convert PSD (Pa2/Hz) to SPL (dB)."""
    psd = torch.clamp(psd, min=1e-40)
    return 10 * torch.log10(psd / (p_ref ** 2))
