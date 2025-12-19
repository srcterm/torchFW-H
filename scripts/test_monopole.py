#!/usr/bin/env python3
"""
Monopole validation test for the FW-H solver.

Verifies the Farassat 1A formulation by comparing numerical results
against the analytic solution for a compact pulsating sphere (monopole).

Analytic Solution:
------------------
For a compact sphere of radius a with uniform oscillating mass flux
Q = Q₀ sin(ωτ) on its surface, the far-field pressure is:

    p'(r, t) = (a² Q₀ ω / r) cos(ω(t - r/c₀))

This test verifies:
1. 1/r pressure decay with distance
2. Amplitude matches analytic solution
3. Results are independent of temporal resolution (dt)
4. Correct retarded time (phase)

Run with: python scripts/test_monopole.py
"""

import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch import Tensor

from src.surfaces.parametric import sphere, PermeableSurface
from src.solver.source_terms import FWHSourceTerms
from src.solver.emission import (
    compute_observer_time_grid,
    compute_emission_context,
    compute_fwh_kernel,
    emission_loop,
)


# =============================================================================
# Test utilities
# =============================================================================

def run_test(test_fn):
    """Run a test function and report result."""
    name = test_fn.__name__
    try:
        test_fn()
        print(f"  ✓ {name}")
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_monopole_source_terms(
    surface: PermeableSurface,
    times: Tensor,
    Q0: float,
    omega: float,
    dt: float
) -> FWHSourceTerms:
    """
    Create synthetic source terms for a pulsating sphere (monopole).

    Q = Q₀ sin(ωτ)  - uniform over surface
    L = 0           - pure monopole (no dipole)

    Args:
        surface: Spherical FW-H surface
        times: (N_t,) source times
        Q0: Mass flux amplitude (kg/m²/s)
        omega: Angular frequency (rad/s)
        dt: Timestep

    Returns:
        FWHSourceTerms with synthetic monopole source
    """
    N_s = surface.n_points
    N_t = times.shape[0]
    device = surface.points.device

    # Time grid: (1, N_t) for broadcasting
    tau = times.unsqueeze(0)  # (1, N_t)

    # Q = Q₀ sin(ωτ) - uniform over surface
    Q = Q0 * torch.sin(omega * tau).expand(N_s, -1)  # (N_s, N_t)

    # Q̇ = Q₀ ω cos(ωτ)
    Q_dot = Q0 * omega * torch.cos(omega * tau).expand(N_s, -1)  # (N_s, N_t)

    # L = 0 (pure monopole)
    L = torch.zeros(N_s, N_t, 3, device=device)
    L_dot = torch.zeros(N_s, N_t, 3, device=device)

    # Velocity and Mach number (not used for stationary surface, but required by dataclass)
    u = torch.zeros(N_s, N_t, 3, device=device)
    M_sq = torch.zeros(N_s, N_t, device=device)

    return FWHSourceTerms(
        Q=Q,
        Q_dot=Q_dot,
        L=L,
        L_dot=L_dot,
        u=u,
        M_sq=M_sq
    )


def analytic_monopole_pressure(
    r: float,
    t: Tensor,
    a: float,
    Q0: float,
    omega: float,
    c0: float
) -> Tensor:
    """
    Analytic far-field pressure for a compact pulsating sphere.

    p'(r, t) = (a² Q₀ ω / r) cos(ω(t - r/c₀))

    For a surface integral of Q̇/r over a sphere of radius a with uniform Q̇:
        ∫∫ Q̇/r dS ≈ (4πa²/r) Q̇  (for compact source, r >> a)
        p' = (1/4π) * (4πa²/r) * Q̇ = (a²/r) * Q̇
        p' = (a² Q₀ ω / r) cos(ω(t - r/c₀))

    Args:
        r: Observer distance from center
        t: Observer times
        a: Sphere radius
        Q0: Mass flux amplitude
        omega: Angular frequency
        c0: Speed of sound

    Returns:
        Analytic pressure signal at observer
    """
    retarded_phase = omega * (t - r / c0)
    return (a**2 * Q0 * omega / r) * torch.cos(retarded_phase)


def run_monopole_solver(
    a: float,
    Q0: float,
    freq: float,
    c0: float,
    observer_distance: float,
    n_periods: int = 4,
    samples_per_period: int = 64,
    n_theta: int = 32,
    n_phi: int = 16
) -> tuple:
    """
    Run the FW-H solver for a monopole test case.

    Args:
        a: Sphere radius
        Q0: Mass flux amplitude
        freq: Frequency in Hz
        c0: Speed of sound
        observer_distance: Distance from center to observer
        n_periods: Number of periods to simulate
        samples_per_period: Timesteps per period
        n_theta, n_phi: Surface resolution

    Returns:
        (observer_times, numerical_pressure, analytic_pressure)
    """
    omega = 2 * math.pi * freq
    period = 1.0 / freq
    duration = n_periods * period
    dt = period / samples_per_period
    N_t = int(duration / dt) + 1

    # Create sphere surface
    surface = sphere(radius=a, center=(0, 0, 0), n_theta=n_theta, n_phi=n_phi)

    # Source time grid
    source_times = torch.linspace(0, duration, N_t)

    # Create synthetic source terms
    source_terms = create_monopole_source_terms(
        surface=surface,
        times=source_times,
        Q0=Q0,
        omega=omega,
        dt=dt
    )

    # Single observer on x-axis
    observer_points = torch.tensor([[observer_distance, 0.0, 0.0]])

    # Compute observer time grid
    observer_times, obs_dt, t_offset = compute_observer_time_grid(
        source_times=source_times,
        surface_points=surface.points,
        observer_points=observer_points,
        c0=c0
    )

    # Compute emission context
    context = compute_emission_context(
        surface_points=surface.points,
        observer_points=observer_points,
        normals=surface.normals
    )

    # Run emission loop
    signal = emission_loop(
        source_terms=source_terms,
        context=context,
        weights=surface.weights,
        source_times=source_times,
        observer_times=observer_times,
        c0=c0,
        dt=dt
    )

    # Apply 1/(4π) normalization
    pressure = signal / (4.0 * math.pi)

    # Analytic solution
    analytic = analytic_monopole_pressure(
        r=observer_distance,
        t=observer_times,
        a=a,
        Q0=Q0,
        omega=omega,
        c0=c0
    )

    return observer_times, pressure[0], analytic


# =============================================================================
# Tests
# =============================================================================

def test_monopole_amplitude():
    """Test that monopole pressure amplitude matches analytic solution."""
    # Parameters
    a = 0.1        # 10 cm sphere radius
    Q0 = 10.0      # kg/m²/s mass flux amplitude
    freq = 100.0   # 100 Hz
    c0 = 340.0     # m/s
    r = 10.0       # 10 m observer distance

    # Compactness check: ka << 1
    k = 2 * math.pi * freq / c0
    ka = k * a
    assert ka < 0.5, f"Source not compact: ka = {ka:.2f}"

    # Run solver
    t_obs, p_num, p_ana = run_monopole_solver(
        a=a, Q0=Q0, freq=freq, c0=c0,
        observer_distance=r,
        n_periods=6,
        samples_per_period=64,
        n_theta=32,
        n_phi=16
    )

    # Skip initial transient (first 2 periods worth)
    travel_time = r / c0
    t_start = travel_time + 2.0 / freq
    mask = t_obs > t_start

    if mask.sum() < 10:
        raise ValueError("Not enough data after transient")

    p_num_trim = p_num[mask]
    p_ana_trim = p_ana[mask]

    # Compare amplitudes (peak values)
    num_amp = p_num_trim.abs().max().item()
    ana_amp = p_ana_trim.abs().max().item()

    rel_error = abs(num_amp - ana_amp) / ana_amp
    assert rel_error < 0.05, f"Amplitude error too large: {rel_error*100:.1f}% (num={num_amp:.4e}, ana={ana_amp:.4e})"


def test_monopole_1_over_r_decay():
    """Test that pressure decays as 1/r with distance."""
    # Parameters
    a = 0.1        # 10 cm sphere
    Q0 = 10.0      # kg/m²/s
    freq = 100.0   # 100 Hz
    c0 = 340.0     # m/s

    # Test at two distances
    r1, r2 = 5.0, 10.0

    _, p1, _ = run_monopole_solver(
        a=a, Q0=Q0, freq=freq, c0=c0,
        observer_distance=r1,
        n_periods=6, samples_per_period=64
    )

    _, p2, _ = run_monopole_solver(
        a=a, Q0=Q0, freq=freq, c0=c0,
        observer_distance=r2,
        n_periods=6, samples_per_period=64
    )

    # RMS pressures (after removing some edge effects)
    trim = len(p1) // 4
    p1_rms = p1[trim:-trim].pow(2).mean().sqrt().item()
    p2_rms = p2[trim:-trim].pow(2).mean().sqrt().item()

    # Expected ratio: p1/p2 = r2/r1
    expected_ratio = r2 / r1
    actual_ratio = p1_rms / p2_rms

    rel_error = abs(actual_ratio - expected_ratio) / expected_ratio
    assert rel_error < 0.05, f"1/r decay error: {rel_error*100:.1f}% (expected {expected_ratio:.2f}, got {actual_ratio:.2f})"


def test_monopole_dt_independence():
    """Test that results don't depend on temporal resolution."""
    # Parameters
    a = 0.1
    Q0 = 10.0
    freq = 100.0
    c0 = 340.0
    r = 10.0

    # Run with two different dt values
    _, p_coarse, _ = run_monopole_solver(
        a=a, Q0=Q0, freq=freq, c0=c0,
        observer_distance=r,
        n_periods=4,
        samples_per_period=32  # Coarser
    )

    _, p_fine, _ = run_monopole_solver(
        a=a, Q0=Q0, freq=freq, c0=c0,
        observer_distance=r,
        n_periods=4,
        samples_per_period=128  # Finer
    )

    # Compare RMS pressures (should be similar regardless of dt)
    trim_coarse = len(p_coarse) // 4
    trim_fine = len(p_fine) // 4

    rms_coarse = p_coarse[trim_coarse:-trim_coarse].pow(2).mean().sqrt().item()
    rms_fine = p_fine[trim_fine:-trim_fine].pow(2).mean().sqrt().item()

    rel_diff = abs(rms_coarse - rms_fine) / rms_fine
    assert rel_diff < 0.05, f"dt dependence detected: {rel_diff*100:.1f}% difference between coarse and fine"


def test_monopole_frequency_content():
    """Test that output contains correct frequency."""
    # Parameters
    a = 0.1
    Q0 = 10.0
    freq = 100.0
    c0 = 340.0
    r = 10.0

    t_obs, p_num, _ = run_monopole_solver(
        a=a, Q0=Q0, freq=freq, c0=c0,
        observer_distance=r,
        n_periods=8,
        samples_per_period=64
    )

    # Compute FFT
    dt = (t_obs[1] - t_obs[0]).item()
    N = len(p_num)
    spectrum = torch.fft.rfft(p_num)
    freqs = torch.fft.rfftfreq(N, dt)

    # Find peak frequency
    power = spectrum.abs()
    peak_idx = power[1:].argmax() + 1  # Skip DC
    peak_freq = freqs[peak_idx].item()

    freq_error = abs(peak_freq - freq) / freq
    assert freq_error < 0.05, f"Peak frequency error: {freq_error*100:.1f}% (expected {freq} Hz, got {peak_freq:.1f} Hz)"


def test_monopole_retarded_time():
    """Test that signal arrives at correct retarded time."""
    # Parameters
    a = 0.1
    Q0 = 10.0
    freq = 50.0  # Lower freq for clearer phase
    c0 = 340.0
    r = 10.0

    t_obs, p_num, p_ana = run_monopole_solver(
        a=a, Q0=Q0, freq=freq, c0=c0,
        observer_distance=r,
        n_periods=6,
        samples_per_period=64
    )

    # Expected travel time
    expected_delay = r / c0

    # Find first significant signal arrival in numerical result
    threshold = 0.1 * p_num.abs().max()
    signal_start_idx = (p_num.abs() > threshold).nonzero(as_tuple=True)[0]

    if len(signal_start_idx) == 0:
        raise ValueError("No significant signal detected")

    numerical_arrival = t_obs[signal_start_idx[0]].item()

    # The signal should arrive around t = r/c0
    # Allow some tolerance for numerical dispersion
    arrival_error = abs(numerical_arrival - expected_delay)
    assert arrival_error < 0.5 / freq, f"Arrival time error: {arrival_error:.4f}s (expected ~{expected_delay:.4f}s)"


def test_kernel_shapes():
    """Test that kernel computation produces correct shapes."""
    N_s, N_o = 100, 5

    # Create dummy inputs
    Q_dot = torch.randn(N_s)
    L = torch.randn(N_s, 3)
    L_dot = torch.randn(N_s, 3)

    # Create dummy context
    from src.solver.emission import EmissionContext
    context = EmissionContext(
        r_vec=torch.randn(N_s, N_o, 3),
        r_mag=torch.rand(N_s, N_o) + 1.0,
        r_inv=torch.rand(N_s, N_o),
        r_inv_sq=torch.rand(N_s, N_o),
        cos_theta=torch.randn(N_s, N_o)
    )
    context.r_vec = context.r_vec / context.r_vec.norm(dim=-1, keepdim=True)

    c0 = 340.0

    kernel = compute_fwh_kernel(Q_dot, L, L_dot, context, c0)

    assert kernel.shape == (N_s, N_o), f"Kernel shape mismatch: {kernel.shape} vs ({N_s}, {N_o})"


def test_pure_loading_term():
    """Test that loading term works correctly (dipole-only case)."""
    # For a dipole: L ≠ 0, Q = 0
    # Far-field: p' ~ (L̇·r̂)/(c₀r)
    # Near-field: p' ~ (L·r̂)/r²

    from src.solver.emission import EmissionContext

    N_s, N_o = 1, 1
    c0 = 340.0

    # Unit vector pointing in x direction
    r_vec = torch.zeros(N_s, N_o, 3)
    r_vec[0, 0, 0] = 1.0

    r = 10.0  # distance
    context = EmissionContext(
        r_vec=r_vec,
        r_mag=torch.tensor([[r]]),
        r_inv=torch.tensor([[1.0/r]]),
        r_inv_sq=torch.tensor([[1.0/r**2]]),
        cos_theta=torch.tensor([[1.0]])
    )

    # Zero thickness, loading in x direction
    Q_dot = torch.zeros(N_s)
    L = torch.tensor([[100.0, 0.0, 0.0]])  # 100 Pa loading in x
    L_dot = torch.tensor([[1000.0, 0.0, 0.0]])  # 1000 Pa/s loading derivative

    kernel = compute_fwh_kernel(Q_dot, L, L_dot, context, c0)

    # Expected:
    # thickness = 0
    # loading_far = (L̇·r̂)/(c₀r) = 1000 / (340 * 10) = 0.294
    # loading_near = (L·r̂)/r² = 100 / 100 = 1.0
    expected_far = 1000.0 / (c0 * r)
    expected_near = 100.0 / r**2
    expected_total = expected_far + expected_near

    assert abs(kernel.item() - expected_total) < 1e-5, \
        f"Loading term error: got {kernel.item():.6f}, expected {expected_total:.6f}"


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("Monopole Validation Tests for FW-H Solver")
    print("=" * 60)

    print("\n[Unit tests]")
    unit_tests = [
        test_kernel_shapes,
        test_pure_loading_term,
    ]
    unit_passed = sum(run_test(t) for t in unit_tests)
    print(f"\nUnit tests: {unit_passed}/{len(unit_tests)} passed")

    print("\n[Physics validation tests]")
    physics_tests = [
        test_monopole_amplitude,
        test_monopole_1_over_r_decay,
        test_monopole_dt_independence,
        test_monopole_frequency_content,
        test_monopole_retarded_time,
    ]
    physics_passed = sum(run_test(t) for t in physics_tests)
    print(f"\nPhysics tests: {physics_passed}/{len(physics_tests)} passed")

    # Summary
    total = len(unit_tests) + len(physics_tests)
    passed = unit_passed + physics_passed
    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
