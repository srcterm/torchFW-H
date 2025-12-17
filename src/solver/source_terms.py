"""
FW-H source term computation.

Converts interpolated CFD fields (rho, u, v, w, p) on the permeable surface
into the source terms Q and L_i needed for the FW-H integrals.

Source terms for porous FW-H (stationary surface):
    Q = rho * u_n           (mass flux through surface)
    L_i = p' * n_i + rho * u_i * u_n    (loading vector)

where:
    u_n = u_i * n_i         (velocity normal to surface)
    p' = p - p0             (pressure fluctuation)
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor

from .derivatives import spectral_derivative


@dataclass
class FWHSourceTerms:
    """
    Container for FW-H source terms on the permeable surface.

    All tensors have shape (N_surface, N_timesteps) or (N_surface, N_timesteps, 3).
    Time axis is axis=1 for consistency with derivative functions.

    Attributes:
        Q: (N_s, N_t) mass flux through surface: rho * u_n
        Q_dot: (N_s, N_t) time derivative of Q
        L: (N_s, N_t, 3) loading vector: p'*n + rho*u*u_n
        L_dot: (N_s, N_t, 3) time derivative of L
        u: (N_s, N_t, 3) velocity vector (needed for M_r computation)
        M_sq: (N_s, N_t) Mach number squared: |u|^2 / c0^2
    """
    Q: Tensor           # (N_s, N_t) mass flux
    Q_dot: Tensor       # (N_s, N_t) time derivative of Q
    L: Tensor           # (N_s, N_t, 3) loading vector
    L_dot: Tensor       # (N_s, N_t, 3) time derivative of L
    u: Tensor           # (N_s, N_t, 3) velocity vector
    M_sq: Tensor        # (N_s, N_t) Mach number squared

    def to(self, device: torch.device) -> 'FWHSourceTerms':
        """Transfer all tensors to specified device."""
        return FWHSourceTerms(
            Q=self.Q.to(device),
            Q_dot=self.Q_dot.to(device),
            L=self.L.to(device),
            L_dot=self.L_dot.to(device),
            u=self.u.to(device),
            M_sq=self.M_sq.to(device),
        )

    @property
    def n_surface(self) -> int:
        """Number of surface points."""
        return self.Q.shape[0]

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps."""
        return self.Q.shape[1]

    @property
    def device(self) -> torch.device:
        """Device where tensors are stored."""
        return self.Q.device


def compute_source_terms(
    rho: Tensor,
    u: Tensor,
    v: Tensor,
    w: Tensor,
    p: Tensor,
    normals: Tensor,
    dt: float,
    c0: float,
    rho0: float,
    p0: Optional[Union[float, Tensor]] = None,
    f_max: Optional[float] = None,
) -> FWHSourceTerms:
    """
    Compute FW-H source terms from interpolated surface fields.

    Args:
        rho: (N_s, N_t) density at surface points over time
        u, v, w: (N_s, N_t) velocity components at surface points
        p: (N_s, N_t) pressure at surface points
        normals: (N_s, 3) outward unit normal vectors (time-invariant)
        dt: timestep in seconds
        c0: speed of sound (m/s)
        rho0: reference density (kg/m³) - not used in current formulation
              but kept for API consistency
        p0: reference pressure for computing p' = p - p0
            - If None: uses time-mean pressure at each point
            - If float: uses that constant value everywhere
            - If Tensor (N_s,): uses per-point reference pressure
        f_max: cutoff frequency for derivative filtering (Hz)
               If None, no filtering is applied (use for clean data only)

    Returns:
        FWHSourceTerms dataclass containing Q, Q_dot, L, L_dot, u, M_sq

    Note:
        All operations preserve gradient flow for autodiff compatibility.
    """
    N_s, N_t = rho.shape
    device = rho.device
    dtype = rho.dtype

    # Validate input shapes
    assert u.shape == (N_s, N_t), f"u shape mismatch: {u.shape} vs ({N_s}, {N_t})"
    assert v.shape == (N_s, N_t), f"v shape mismatch: {v.shape} vs ({N_s}, {N_t})"
    assert w.shape == (N_s, N_t), f"w shape mismatch: {w.shape} vs ({N_s}, {N_t})"
    assert p.shape == (N_s, N_t), f"p shape mismatch: {p.shape} vs ({N_s}, {N_t})"
    assert normals.shape == (N_s, 3), f"normals shape mismatch: {normals.shape} vs ({N_s}, 3)"

    # Build velocity vector: (N_s, N_t, 3)
    vel = torch.stack([u, v, w], dim=-1)

    # Normal velocity: u_n = u_i * n_i
    # normals: (N_s, 3) -> (N_s, 1, 3) for broadcasting with vel: (N_s, N_t, 3)
    u_n = (vel * normals.unsqueeze(1)).sum(dim=-1)  # (N_s, N_t)

    # Pressure fluctuation: p' = p - p0
    if p0 is None:
        # Use time-mean at each point
        p0_value = p.mean(dim=1, keepdim=True)  # (N_s, 1)
    elif isinstance(p0, (int, float)):
        # Scalar reference
        p0_value = torch.tensor(p0, device=device, dtype=dtype)
    else:
        # Tensor reference - ensure proper shape
        p0_value = p0.to(device=device, dtype=dtype)
        if p0_value.dim() == 1:
            p0_value = p0_value.unsqueeze(1)  # (N_s,) -> (N_s, 1)

    p_prime = p - p0_value  # (N_s, N_t)

    # Mass flux: Q = rho * u_n
    Q = rho * u_n  # (N_s, N_t)

    # Loading vector: L_i = p' * n_i + rho * u_i * u_n
    # p' * n: (N_s, N_t, 1) * (N_s, 1, 3) -> (N_s, N_t, 3)
    # rho * u * u_n: (N_s, N_t, 1) * (N_s, N_t, 3) * (N_s, N_t, 1) -> (N_s, N_t, 3)
    L = (
        p_prime.unsqueeze(-1) * normals.unsqueeze(1) +
        rho.unsqueeze(-1) * vel * u_n.unsqueeze(-1)
    )  # (N_s, N_t, 3)

    # Mach number squared: M^2 = |u|^2 / c0^2
    vel_sq = (vel ** 2).sum(dim=-1)  # (N_s, N_t)
    M_sq = vel_sq / (c0 ** 2)  # (N_s, N_t)

    # Time derivatives with optional filtering
    Q_dot = spectral_derivative(Q, dt, f_max=f_max, axis=1)  # (N_s, N_t)
    L_dot = spectral_derivative(L, dt, f_max=f_max, axis=1)  # (N_s, N_t, 3)

    return FWHSourceTerms(
        Q=Q,
        Q_dot=Q_dot,
        L=L,
        L_dot=L_dot,
        u=vel,
        M_sq=M_sq,
    )


def compute_pressure_fluctuation(
    p: Tensor,
    p0: Optional[Union[float, Tensor]] = None,
) -> Tensor:
    """
    Compute pressure fluctuation p' = p - p0.

    Utility function for cases where only pressure fluctuation is needed.

    Args:
        p: (N_s, N_t) pressure field
        p0: reference pressure (None for time-mean, scalar, or per-point tensor)

    Returns:
        (N_s, N_t) pressure fluctuation
    """
    device = p.device
    dtype = p.dtype

    if p0 is None:
        p0_value = p.mean(dim=1, keepdim=True)
    elif isinstance(p0, (int, float)):
        p0_value = torch.tensor(p0, device=device, dtype=dtype)
    else:
        p0_value = p0.to(device=device, dtype=dtype)
        if p0_value.dim() == 1:
            p0_value = p0_value.unsqueeze(1)

    return p - p0_value
