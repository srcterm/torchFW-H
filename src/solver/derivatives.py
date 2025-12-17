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

    Args:
        signal: (..., N_t) time series, must be uniformly sampled
        dt: time step in seconds
        f_max: cutoff frequency in Hz. Content above this is attenuated
               before differentiation. If None, no filtering (use only
               for clean synthetic data).
        axis: axis along which to differentiate (default: -1, last axis)

    Returns:
        (..., N_t) time derivative

    Note:
        Maintains gradient flow for autodiff. Avoids in-place operations.
    """
    # Move axis to last position for easier handling
    signal = torch.moveaxis(signal, axis, -1)
    original_shape = signal.shape
    N = signal.shape[-1]
    device = signal.device
    dtype = signal.dtype

    # Frequency array for rfft (only positive frequencies)
    freq = torch.fft.rfftfreq(N, dt, device=device)

    # Forward FFT (real-to-complex)
    spectrum = torch.fft.rfft(signal, dim=-1)

    # Apply low-pass filter if requested
    if f_max is not None:
        spectrum = _apply_lowpass(spectrum, freq, f_max)

    # Differentiation in frequency domain: multiply by i*omega
    # omega = 2*pi*f, and i*omega for derivative
    omega = 2j * torch.pi * freq.to(dtype=torch.complex64)

    # Broadcast omega to match spectrum shape
    # spectrum has shape (..., N_freq), omega has shape (N_freq,)
    spectrum = spectrum * omega

    # Inverse FFT back to time domain
    derivative = torch.fft.irfft(spectrum, n=N, dim=-1)

    # Ensure output dtype matches input (irfft returns float)
    if derivative.dtype != dtype:
        derivative = derivative.to(dtype)

    # Move axis back to original position
    derivative = torch.moveaxis(derivative, -1, axis)

    return derivative


def _apply_lowpass(
    spectrum: Tensor,
    freq: Tensor,
    f_max: float,
    rolloff_fraction: float = 0.1
) -> Tensor:
    """
    Apply smooth low-pass filter with raised-cosine rolloff.

    Transition band is rolloff_fraction * f_max wide, centered at f_max.
    This avoids Gibbs ringing artifacts from hard cutoffs.

    Args:
        spectrum: (..., N_freq) complex spectrum from rfft
        freq: (N_freq,) frequency array from rfftfreq
        f_max: cutoff frequency in Hz
        rolloff_fraction: width of transition band as fraction of f_max

    Returns:
        (..., N_freq) filtered spectrum
    """
    rolloff = rolloff_fraction * f_max
    f_low = f_max - rolloff
    f_high = f_max + rolloff

    # Smooth transition using raised cosine
    # Below f_low: pass (1.0)
    # Above f_high: block (0.0)
    # Between: smooth transition
    mask = torch.where(
        freq <= f_low,
        torch.ones_like(freq),
        torch.where(
            freq >= f_high,
            torch.zeros_like(freq),
            0.5 * (1.0 + torch.cos(torch.pi * (freq - f_low) / (2 * rolloff)))
        )
    )

    # Apply mask (broadcasts over batch dimensions)
    return spectrum * mask


def suggest_f_max(mean_spacing: float, c0: float, points_per_wavelength: float = 6.0) -> float:
    """
    Suggest cutoff frequency based on surface resolution.

    Uses the criterion that we need at least `points_per_wavelength` surface
    points per acoustic wavelength to resolve the integral accurately.

    Args:
        mean_spacing: mean distance between surface points (meters)
        c0: speed of sound (m/s)
        points_per_wavelength: resolution criterion (default 6)

    Returns:
        Suggested f_max in Hz
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
    Apply low-pass filter to a signal without differentiation.

    Useful for filtering noisy data before other processing.

    Args:
        signal: (..., N_t) time series
        dt: time step in seconds
        f_max: cutoff frequency in Hz
        axis: axis along which to filter

    Returns:
        (..., N_t) filtered signal
    """
    # Move axis to last position
    signal = torch.moveaxis(signal, axis, -1)
    N = signal.shape[-1]
    device = signal.device
    dtype = signal.dtype

    # Frequency array
    freq = torch.fft.rfftfreq(N, dt, device=device)

    # Forward FFT
    spectrum = torch.fft.rfft(signal, dim=-1)

    # Apply filter
    spectrum = _apply_lowpass(spectrum, freq, f_max)

    # Inverse FFT
    filtered = torch.fft.irfft(spectrum, n=N, dim=-1)

    if filtered.dtype != dtype:
        filtered = filtered.to(dtype)

    # Move axis back
    filtered = torch.moveaxis(filtered, -1, axis)

    return filtered
