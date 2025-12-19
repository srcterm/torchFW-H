#!/usr/bin/env python3
"""
Unit tests for Stage 2A: derivatives.py and source_terms.py

Run with: python scripts/test_stage2a.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import math

from src.solver.derivatives import (
    spectral_derivative,
    suggest_f_max,
    apply_lowpass_filter,
    _apply_lowpass,
)
from src.solver.source_terms import (
    FWHSourceTerms,
    compute_source_terms,
    compute_pressure_fluctuation,
)


# =============================================================================
# Test utilities
# =============================================================================

def assert_close(a, b, rtol=1e-3, atol=1e-6, msg=""):
    """Assert two tensors are close."""
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        max_diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg} Max diff: {max_diff}, rtol={rtol}, atol={atol}")


def assert_close_signal(a, b, rel_tol=0.01, msg=""):
    """
    Assert two signal tensors are close using peak-relative tolerance.

    More appropriate for signals that cross zero, where element-wise
    rtol doesn't work well near zero crossings.
    """
    max_diff = (a - b).abs().max().item()
    peak = max(a.abs().max().item(), b.abs().max().item(), 1e-10)
    rel_error = max_diff / peak
    if rel_error > rel_tol:
        raise AssertionError(f"{msg} Rel error: {rel_error:.4f} (max_diff={max_diff:.6f}, peak={peak:.2f})")


def run_test(test_fn):
    """Run a test function and report result."""
    name = test_fn.__name__
    try:
        test_fn()
        print(f"  ✓ {name}")
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return False


# =============================================================================
# Tests for derivatives.py
# =============================================================================

def test_derivative_sine():
    """Derivative of sin(wt) should be w*cos(wt)."""
    dt = 1e-4
    N = 1000
    t = torch.arange(N) * dt
    f = 100.0  # Hz
    omega = 2 * math.pi * f

    signal = torch.sin(omega * t)
    deriv = spectral_derivative(signal, dt)
    expected = omega * torch.cos(omega * t)

    # Trim edges where FFT periodicity causes artifacts
    trim = 50
    # Use peak-relative tolerance since signal crosses zero
    assert_close_signal(
        deriv[trim:-trim],
        expected[trim:-trim],
        rel_tol=0.001,  # 0.1% of peak amplitude
        msg="Sine derivative mismatch"
    )


def test_derivative_constant():
    """Derivative of constant should be near zero."""
    dt = 1e-4
    N = 1000
    signal = torch.ones(N) * 5.0

    deriv = spectral_derivative(signal, dt)

    # FFT of constant has small numerical noise, allow small tolerance
    assert deriv.abs().max() < 0.01, f"Constant derivative not near zero: {deriv.abs().max()}"


def test_derivative_linear():
    """Derivative of linear function should be constant."""
    dt = 1e-4
    N = 1000
    t = torch.arange(N) * dt
    slope = 3.5

    # Linear signal has edge discontinuity for FFT (not periodic)
    # Use a windowed or periodic-friendly test instead
    signal = slope * t

    deriv = spectral_derivative(signal, dt)

    # Trim edges heavily due to non-periodic signal
    trim = 200
    expected = torch.ones(N) * slope

    # Check that mean derivative is close to slope
    mean_deriv = deriv[trim:-trim].mean().item()
    assert abs(mean_deriv - slope) / slope < 0.05, f"Mean derivative mismatch: {mean_deriv} vs {slope}"


def test_derivative_filtering():
    """High frequency content should be attenuated when f_max is set."""
    dt = 1e-4
    N = 1000
    t = torch.arange(N) * dt

    # Signal with low and high frequency components
    f_low, f_high = 100.0, 2000.0
    signal = torch.sin(2 * math.pi * f_low * t) + torch.sin(2 * math.pi * f_high * t)

    # Derivative with f_max between the two frequencies
    f_max = 500.0
    deriv = spectral_derivative(signal, dt, f_max=f_max)

    # Expected: derivative of low-freq component only (high-freq filtered out)
    expected = 2 * math.pi * f_low * torch.cos(2 * math.pi * f_low * t)

    # Should match low-freq derivative (high-freq attenuated)
    trim = 100
    # Use peak-relative tolerance
    assert_close_signal(
        deriv[trim:-trim],
        expected[trim:-trim],
        rel_tol=0.15,  # Allow some tolerance due to rolloff
        msg="Filtered derivative mismatch"
    )


def test_derivative_batch():
    """Verify batched input produces correct output shape."""
    dt = 1e-4
    N_s, N_t = 100, 500
    signal = torch.randn(N_s, N_t)

    deriv = spectral_derivative(signal, dt, axis=1)

    assert deriv.shape == (N_s, N_t), f"Shape mismatch: {deriv.shape} vs ({N_s}, {N_t})"


def test_derivative_preserves_grad():
    """Autodiff should work through spectral_derivative."""
    dt = 1e-4
    N = 100
    signal = torch.randn(N, requires_grad=True)

    deriv = spectral_derivative(signal, dt)
    loss = deriv.sum()
    loss.backward()

    assert signal.grad is not None, "Gradient not computed"
    assert signal.grad.shape == signal.shape, "Gradient shape mismatch"


def test_suggest_f_max():
    """Test f_max suggestion from surface resolution."""
    mean_spacing = 0.01  # 1 cm
    c0 = 340.0  # m/s
    ppw = 6.0  # points per wavelength

    f_max = suggest_f_max(mean_spacing, c0, ppw)

    # f_max = c0 / (spacing * ppw) = 340 / (0.01 * 6) = 5666.67 Hz
    expected = c0 / (mean_spacing * ppw)
    assert abs(f_max - expected) < 1e-6, f"f_max mismatch: {f_max} vs {expected}"


def test_lowpass_filter():
    """Test standalone lowpass filter."""
    dt = 1e-4
    N = 1000
    t = torch.arange(N) * dt

    # Signal with two frequencies
    f_low, f_high = 100.0, 2000.0
    signal = torch.sin(2 * math.pi * f_low * t) + torch.sin(2 * math.pi * f_high * t)

    # Filter
    f_max = 500.0
    filtered = apply_lowpass_filter(signal, dt, f_max)

    # High frequency should be attenuated
    # Check by computing FFT of filtered signal
    spectrum = torch.fft.rfft(filtered)
    freq = torch.fft.rfftfreq(N, dt)

    # Find power at high frequency
    high_idx = (freq - f_high).abs().argmin()
    high_power = spectrum[high_idx].abs()

    # Find power at low frequency
    low_idx = (freq - f_low).abs().argmin()
    low_power = spectrum[low_idx].abs()

    # High should be much smaller than low
    ratio = high_power / low_power
    assert ratio < 0.1, f"High freq not sufficiently attenuated: ratio={ratio}"


# =============================================================================
# Tests for source_terms.py
# =============================================================================

def test_source_terms_shapes():
    """Verify output shapes are correct."""
    N_s, N_t = 100, 50

    rho = torch.ones(N_s, N_t)
    u = torch.zeros(N_s, N_t)
    v = torch.zeros(N_s, N_t)
    w = torch.ones(N_s, N_t)  # flow in z direction
    p = torch.ones(N_s, N_t) * 101325

    normals = torch.zeros(N_s, 3)
    normals[:, 2] = 1.0  # normals in z direction

    terms = compute_source_terms(
        rho, u, v, w, p, normals,
        dt=1e-4, c0=340, rho0=1.225
    )

    assert terms.Q.shape == (N_s, N_t), f"Q shape: {terms.Q.shape}"
    assert terms.Q_dot.shape == (N_s, N_t), f"Q_dot shape: {terms.Q_dot.shape}"
    assert terms.L.shape == (N_s, N_t, 3), f"L shape: {terms.L.shape}"
    assert terms.L_dot.shape == (N_s, N_t, 3), f"L_dot shape: {terms.L_dot.shape}"
    assert terms.u.shape == (N_s, N_t, 3), f"u shape: {terms.u.shape}"
    assert terms.M_sq.shape == (N_s, N_t), f"M_sq shape: {terms.M_sq.shape}"


def test_source_terms_steady_flow():
    """For steady flow, Q_dot and L_dot should be near zero."""
    N_s, N_t = 100, 50

    # Constant fields (steady flow)
    rho = torch.ones(N_s, N_t) * 1.225
    u = torch.ones(N_s, N_t) * 10.0
    v = torch.zeros(N_s, N_t)
    w = torch.zeros(N_s, N_t)
    p = torch.ones(N_s, N_t) * 101325

    normals = torch.zeros(N_s, 3)
    normals[:, 0] = 1.0  # normals in x direction

    terms = compute_source_terms(
        rho, u, v, w, p, normals,
        dt=1e-4, c0=340, rho0=1.225
    )

    # Time derivatives should be small for steady flow
    # FFT has some numerical noise, so allow reasonable tolerance
    Q_magnitude = terms.Q.abs().max().item()
    L_magnitude = terms.L.abs().max().item()

    Q_dot_rel = terms.Q_dot.abs().max().item() / max(Q_magnitude, 1e-10)
    L_dot_rel = terms.L_dot.abs().max().item() / max(L_magnitude, 1e-10)

    assert Q_dot_rel < 0.01, f"Q_dot too large relative to Q: {Q_dot_rel}"
    assert L_dot_rel < 0.01, f"L_dot too large relative to L: {L_dot_rel}"


def test_source_terms_normal_velocity():
    """Test that Q correctly computes normal velocity."""
    N_s, N_t = 10, 20

    rho = torch.ones(N_s, N_t) * 1.225
    u = torch.ones(N_s, N_t) * 10.0  # 10 m/s in x
    v = torch.zeros(N_s, N_t)
    w = torch.zeros(N_s, N_t)
    p = torch.ones(N_s, N_t) * 101325

    # Normals in x direction -> u_n = u
    normals = torch.zeros(N_s, 3)
    normals[:, 0] = 1.0

    terms = compute_source_terms(
        rho, u, v, w, p, normals,
        dt=1e-4, c0=340, rho0=1.225
    )

    # Q = rho * u_n = 1.225 * 10 = 12.25
    expected_Q = 1.225 * 10.0
    assert_close(
        terms.Q,
        torch.ones_like(terms.Q) * expected_Q,
        rtol=1e-5,
        msg="Q value mismatch"
    )


def test_source_terms_perpendicular_flow():
    """Flow perpendicular to normal should give Q = 0."""
    N_s, N_t = 10, 20

    rho = torch.ones(N_s, N_t) * 1.225
    u = torch.ones(N_s, N_t) * 10.0  # flow in x
    v = torch.zeros(N_s, N_t)
    w = torch.zeros(N_s, N_t)
    p = torch.ones(N_s, N_t) * 101325

    # Normals in z direction (perpendicular to flow)
    normals = torch.zeros(N_s, 3)
    normals[:, 2] = 1.0

    terms = compute_source_terms(
        rho, u, v, w, p, normals,
        dt=1e-4, c0=340, rho0=1.225
    )

    # Q = rho * u_n = rho * (u dot n) = 0
    assert terms.Q.abs().max() < 1e-10, f"Q not zero for perpendicular flow: {terms.Q.abs().max()}"


def test_source_terms_mach_number():
    """Test Mach number squared computation."""
    N_s, N_t = 10, 20
    c0 = 340.0

    rho = torch.ones(N_s, N_t) * 1.225
    u = torch.ones(N_s, N_t) * 34.0   # 34 m/s -> M = 0.1
    v = torch.zeros(N_s, N_t)
    w = torch.zeros(N_s, N_t)
    p = torch.ones(N_s, N_t) * 101325

    normals = torch.zeros(N_s, 3)
    normals[:, 0] = 1.0

    terms = compute_source_terms(
        rho, u, v, w, p, normals,
        dt=1e-4, c0=c0, rho0=1.225
    )

    # M^2 = |u|^2 / c0^2 = 34^2 / 340^2 = 0.01
    expected_M_sq = (34.0 / c0) ** 2
    assert_close(
        terms.M_sq,
        torch.ones_like(terms.M_sq) * expected_M_sq,
        rtol=1e-5,
        msg="M_sq mismatch"
    )


def test_source_terms_p0_options():
    """Test different p0 reference pressure options."""
    N_s, N_t = 10, 20

    rho = torch.ones(N_s, N_t) * 1.225
    u = torch.zeros(N_s, N_t)
    v = torch.zeros(N_s, N_t)
    w = torch.zeros(N_s, N_t)

    # Pressure with some variation
    p = torch.ones(N_s, N_t) * 101325
    p[:, :10] = 101325 + 100  # First half slightly higher

    normals = torch.zeros(N_s, 3)
    normals[:, 0] = 1.0

    # Test p0 = None (time-mean)
    terms1 = compute_source_terms(
        rho, u, v, w, p, normals,
        dt=1e-4, c0=340, rho0=1.225, p0=None
    )
    # L should use p' = p - mean(p)

    # Test p0 = scalar
    terms2 = compute_source_terms(
        rho, u, v, w, p, normals,
        dt=1e-4, c0=340, rho0=1.225, p0=101325.0
    )
    # L should use p' = p - 101325

    # Test p0 = tensor
    p0_tensor = torch.ones(N_s) * 101325
    terms3 = compute_source_terms(
        rho, u, v, w, p, normals,
        dt=1e-4, c0=340, rho0=1.225, p0=p0_tensor
    )

    # All should produce valid results (just check shapes)
    assert terms1.L.shape == (N_s, N_t, 3)
    assert terms2.L.shape == (N_s, N_t, 3)
    assert terms3.L.shape == (N_s, N_t, 3)


def test_source_terms_to_device():
    """Test FWHSourceTerms.to() method."""
    N_s, N_t = 10, 20

    rho = torch.ones(N_s, N_t)
    u = torch.zeros(N_s, N_t)
    v = torch.zeros(N_s, N_t)
    w = torch.zeros(N_s, N_t)
    p = torch.ones(N_s, N_t) * 101325
    normals = torch.zeros(N_s, 3)
    normals[:, 0] = 1.0

    terms = compute_source_terms(
        rho, u, v, w, p, normals,
        dt=1e-4, c0=340, rho0=1.225
    )

    # Move to same device (CPU) - should work
    terms_cpu = terms.to(torch.device('cpu'))
    assert terms_cpu.Q.device.type == 'cpu'
    assert terms_cpu.L.device.type == 'cpu'


def test_source_terms_preserves_grad():
    """Test autodiff compatibility of source terms."""
    N_s, N_t = 10, 20

    # Use leaf tensors with requires_grad
    rho_base = torch.ones(N_s, N_t, requires_grad=True)
    p_base = torch.randn(N_s, N_t, requires_grad=True)  # Non-uniform for gradient flow

    # Scale them (these become non-leaf but gradient flows to base)
    rho = rho_base * 1.225
    u = torch.ones(N_s, N_t) * 10.0
    v = torch.zeros(N_s, N_t)
    w = torch.zeros(N_s, N_t)
    p = p_base * 1000 + 101325  # Add variation around reference
    normals = torch.zeros(N_s, 3)
    normals[:, 0] = 1.0

    # Use explicit p0 so p_prime = p - p0 is non-zero
    terms = compute_source_terms(
        rho, u, v, w, p, normals,
        dt=1e-4, c0=340, rho0=1.225, p0=101325.0
    )

    # Compute loss and backward
    loss = terms.Q.sum() + terms.L.sum()
    loss.backward()

    # Check gradients on leaf tensors
    assert rho_base.grad is not None, "rho gradient not computed"
    assert p_base.grad is not None, "p gradient not computed"
    assert rho_base.grad.abs().sum() > 0, "rho gradient is all zeros"
    assert p_base.grad.abs().sum() > 0, "p gradient is all zeros"


def test_pressure_fluctuation():
    """Test compute_pressure_fluctuation utility."""
    N_s, N_t = 10, 20
    p = torch.ones(N_s, N_t) * 101325
    p[:, :10] += 100  # Add fluctuation

    # p0 = None -> time mean
    p_prime = compute_pressure_fluctuation(p, p0=None)
    assert p_prime.mean().abs() < 1e-5, "Mean of p' should be ~0 with time-mean reference"

    # p0 = scalar
    p_prime = compute_pressure_fluctuation(p, p0=101325.0)
    assert_close(
        p_prime[:, :10],
        torch.ones(N_s, 10) * 100,
        msg="p' mismatch with scalar p0"
    )


# =============================================================================
# Main test runner
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("Stage 2A Unit Tests: derivatives.py and source_terms.py")
    print("=" * 60)

    # Derivatives tests
    print("\n[derivatives.py tests]")
    deriv_tests = [
        test_derivative_sine,
        test_derivative_constant,
        test_derivative_linear,
        test_derivative_filtering,
        test_derivative_batch,
        test_derivative_preserves_grad,
        test_suggest_f_max,
        test_lowpass_filter,
    ]

    deriv_passed = sum(run_test(t) for t in deriv_tests)
    print(f"\nDerivatives: {deriv_passed}/{len(deriv_tests)} passed")

    # Source terms tests
    print("\n[source_terms.py tests]")
    source_tests = [
        test_source_terms_shapes,
        test_source_terms_steady_flow,
        test_source_terms_normal_velocity,
        test_source_terms_perpendicular_flow,
        test_source_terms_mach_number,
        test_source_terms_p0_options,
        test_source_terms_to_device,
        test_source_terms_preserves_grad,
        test_pressure_fluctuation,
    ]

    source_passed = sum(run_test(t) for t in source_tests)
    print(f"\nSource terms: {source_passed}/{len(source_tests)} passed")

    # Summary
    total = len(deriv_tests) + len(source_tests)
    passed = deriv_passed + source_passed
    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
