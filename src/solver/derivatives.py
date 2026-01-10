"""
Time derivative computation with spectral filtering.

Provides stable differentiation for noisy CFD data by applying
a low-pass filter before/during differentiation. Uses FFT-based
spectral methods for accuracy.
"""

import torch
from torch import Tensor


def spectral_derivative(
    signal: Tensor,
    dt: float,
    f_max: float | None = None,
    axis: int = -1
) -> Tensor:
    """
    Compute time derivative using FFT with optional low-pass filter.
    The filter prevents noise amplification at high frequencies where
    d/dt amplifies by factor of omega.
    """

    signal = torch.moveaxis(signal, axis, -1)
    original_shape = signal.shape
    N = signal.shape[-1]
    device = signal.device
    dtype = signal.dtype
    freq = torch.fft.rfftfreq(N, dt, device=device)
    spectrum = torch.fft.rfft(signal, dim=-1)

    # Apply low-pass filter?
    if f_max is not None:
        spectrum = _apply_lowpass(spectrum, freq, f_max)
    omega = 2j * torch.pi * freq.to(dtype=torch.complex64)
    spectrum = spectrum * omega

    derivative = torch.fft.irfft(spectrum, n=N, dim=-1)
    if derivative.dtype != dtype:
        derivative = derivative.to(dtype)
    derivative = torch.moveaxis(derivative, -1, axis)
    return derivative


def _apply_lowpass(
    spectrum: Tensor,
    freq: Tensor,
    f_max: float,
    rolloff_fraction: float = 0.1
) -> Tensor:
    """
    Smooth low-pass filter with cos rolloff.
    """
    rolloff = rolloff_fraction * f_max
    f_low = f_max - rolloff
    f_high = f_max + rolloff

    mask = torch.where(
        freq <= f_low,
        torch.ones_like(freq),
        torch.where(
            freq >= f_high,
            torch.zeros_like(freq),
            0.5 * (1.0 + torch.cos(torch.pi * (freq - f_low) / (2 * rolloff)))
        )
    )
    return spectrum * mask


def suggest_f_max(mean_spacing: float, c0: float, points_per_wavelength: float = 6.0) -> float:
    """
    Suggest cutoff frequency based on surface resolution.
    Uses the criterion that we need at least `points_per_wavelength` surface
    points per acoustic wavelength to resolve the integral accurately.
    """
    # lambda_min = mean_spacing * points_per_wavelength
    # f_max = c0 / lambda_min
    lambda_min = mean_spacing * points_per_wavelength
    return c0 / lambda_min


def apply_lowpass_filter(
    signal: Tensor,
    dt: float,
    f_max: float,
    axis: int = -1
) -> Tensor:
    """
    Low-pass filter to a signal without differentiation.
    Data filter for preprocessing out numerical nosie.
    """
    signal = torch.moveaxis(signal, axis, -1)
    N = signal.shape[-1]
    device = signal.device
    dtype = signal.dtype
    freq = torch.fft.rfftfreq(N, dt, device=device)
    spectrum = torch.fft.rfft(signal, dim=-1)
    spectrum = _apply_lowpass(spectrum, freq, f_max)
    filtered = torch.fft.irfft(spectrum, n=N, dim=-1)

    if filtered.dtype != dtype:
        filtered = filtered.to(dtype)
    filtered = torch.moveaxis(filtered, -1, axis)
    return filtered
