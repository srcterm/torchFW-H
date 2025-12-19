"""
FW-H emission computation and time-domain accumulation.

Implements the Farassat 1A formulation for a stationary permeable FW-H surface:
- Compute observer time grid based on source-observer distances
- Compute FW-H integrand (kernel) at each source time
- Accumulate emissions into observer time bins with linear interpolation

The key equation for a stationary surface is:

    p'(x, t) = (1/4π) ∫∫_S [ Q̇/r + (L̇·r̂)/(c₀r) + (L·r̂)/r² ] dS

Where:
    - Thickness (monopole): Q̇/r
    - Loading far-field:    (L̇·r̂)/(c₀r)  — 1/r decay
    - Loading near-field:   (L·r̂)/r²     — 1/r² decay

The retarded time relation is:
    t = τ + r/c₀  (observer time = source time + travel time)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .source_terms import FWHSourceTerms


def compute_observer_time_grid(
    source_times: Tensor,
    surface_points: Tensor,
    observer_points: Tensor,
    c0: float,
    n_pad: int = 10
) -> Tuple[Tensor, float, float]:
    """
    Compute the observer time grid based on source times and distances.

    The observer time range is determined by:
    - Earliest arrival: first source time + min distance / c0
    - Latest arrival: last source time + max distance / c0

    Uses the same dt as the source time grid.

    Args:
        source_times: (N_t,) source emission times
        surface_points: (N_s, 3) FW-H surface point positions
        observer_points: (N_o, 3) observer positions
        c0: speed of sound (m/s)
        n_pad: number of extra timesteps to pad at start/end

    Returns:
        observer_times: (N_t_obs,) observer time grid
        dt: timestep (same as source)
        t_offset: offset of first observer time from first source time
    """
    # Compute source timestep
    N_t = source_times.shape[0]
    if N_t < 2:
        raise ValueError("Need at least 2 source timesteps")

    dt = (source_times[1] - source_times[0]).item()

    # Compute min/max distances from any surface point to any observer
    # For memory efficiency, we don't form the full distance matrix
    # Instead compute bounds from bounding boxes

    # Surface bounds
    s_min = surface_points.min(dim=0).values  # (3,)
    s_max = surface_points.max(dim=0).values  # (3,)

    # Observer bounds
    o_min = observer_points.min(dim=0).values  # (3,)
    o_max = observer_points.max(dim=0).values  # (3,)

    # Min distance: closest corners of bounding boxes
    # Max distance: farthest corners of bounding boxes

    # For more accurate bounds, sample a few points
    # Use centroid + corners approach
    s_centroid = surface_points.mean(dim=0)
    o_centroid = observer_points.mean(dim=0)

    # Sample distances: centroid to centroid and corner extremes
    centroid_dist = torch.linalg.norm(s_centroid - o_centroid).item()

    # Estimate min/max from bounding box corners
    corners_s = torch.stack([
        torch.tensor([s_min[0], s_min[1], s_min[2]]),
        torch.tensor([s_min[0], s_min[1], s_max[2]]),
        torch.tensor([s_min[0], s_max[1], s_min[2]]),
        torch.tensor([s_min[0], s_max[1], s_max[2]]),
        torch.tensor([s_max[0], s_min[1], s_min[2]]),
        torch.tensor([s_max[0], s_min[1], s_max[2]]),
        torch.tensor([s_max[0], s_max[1], s_min[2]]),
        torch.tensor([s_max[0], s_max[1], s_max[2]]),
    ])  # (8, 3)

    corners_o = torch.stack([
        torch.tensor([o_min[0], o_min[1], o_min[2]]),
        torch.tensor([o_min[0], o_min[1], o_max[2]]),
        torch.tensor([o_min[0], o_max[1], o_min[2]]),
        torch.tensor([o_min[0], o_max[1], o_max[2]]),
        torch.tensor([o_max[0], o_min[1], o_min[2]]),
        torch.tensor([o_max[0], o_min[1], o_max[2]]),
        torch.tensor([o_max[0], o_max[1], o_min[2]]),
        torch.tensor([o_max[0], o_max[1], o_max[2]]),
    ])  # (8, 3)

    # Distance matrix between corners (small: 8x8)
    corner_dists = torch.cdist(corners_s, corners_o)
    r_min = corner_dists.min().item()
    r_max = corner_dists.max().item()

    # Ensure r_min is not zero (would cause issues)
    r_min = max(r_min, 1e-6)

    # Observer time range
    tau_start = source_times[0].item()
    tau_end = source_times[-1].item()

    t_start = tau_start + r_min / c0 - n_pad * dt
    t_end = tau_end + r_max / c0 + n_pad * dt

    # Create observer time grid
    N_t_obs = int((t_end - t_start) / dt) + 1
    observer_times = torch.linspace(t_start, t_start + (N_t_obs - 1) * dt, N_t_obs)

    t_offset = t_start - tau_start

    return observer_times, dt, t_offset


@dataclass
class EmissionContext:
    """
    Pre-computed quantities for emission accumulation.

    These are computed once and reused for all source timesteps.

    Attributes:
        r_vec: (N_s, N_o, 3) unit vectors from surface to observers
        r_mag: (N_s, N_o) distances from surface to observers
        r_inv: (N_s, N_o) 1/r for far-field term
        r_inv_sq: (N_s, N_o) 1/r² for near-field term
        cos_theta: (N_s, N_o) cos of angle between normal and r_vec
    """
    r_vec: Tensor       # (N_s, N_o, 3) unit vector from source to observer
    r_mag: Tensor       # (N_s, N_o) distance
    r_inv: Tensor       # (N_s, N_o) 1/r
    r_inv_sq: Tensor    # (N_s, N_o) 1/r²
    cos_theta: Tensor   # (N_s, N_o) normal · r_hat


def compute_emission_context(
    surface_points: Tensor,
    observer_points: Tensor,
    normals: Tensor,
    r_min_clamp: float = 1e-6
) -> EmissionContext:
    """
    Pre-compute geometric quantities for emission accumulation.

    Args:
        surface_points: (N_s, 3) FW-H surface positions
        observer_points: (N_o, 3) observer positions
        normals: (N_s, 3) surface outward normals
        r_min_clamp: minimum distance to avoid 1/r singularity

    Returns:
        EmissionContext with pre-computed geometric factors
    """
    N_s = surface_points.shape[0]
    N_o = observer_points.shape[0]
    device = surface_points.device

    # Vector from each surface point to each observer: (N_s, N_o, 3)
    # r_vec[i, j] = observer[j] - surface[i]
    r_vec = observer_points.unsqueeze(0) - surface_points.unsqueeze(1)  # (N_s, N_o, 3)

    # Distance magnitude
    r_mag = torch.linalg.norm(r_vec, dim=-1)  # (N_s, N_o)
    r_mag = torch.clamp(r_mag, min=r_min_clamp)  # Avoid division by zero

    # Unit direction vector
    r_hat = r_vec / r_mag.unsqueeze(-1)  # (N_s, N_o, 3)

    # 1/r and 1/r² terms
    r_inv = 1.0 / r_mag  # (N_s, N_o)
    r_inv_sq = r_inv * r_inv  # (N_s, N_o)

    # cos(theta) = n · r_hat, where n is surface normal
    # normals: (N_s, 3) -> (N_s, 1, 3)
    # r_hat: (N_s, N_o, 3)
    cos_theta = (normals.unsqueeze(1) * r_hat).sum(dim=-1)  # (N_s, N_o)

    return EmissionContext(
        r_vec=r_hat,  # Store unit vector, not raw vector
        r_mag=r_mag,
        r_inv=r_inv,
        r_inv_sq=r_inv_sq,
        cos_theta=cos_theta
    )


def compute_emission_context_chunked(
    surface_points: Tensor,
    observer_points: Tensor,
    normals: Tensor,
    device: torch.device,
    chunk_size: int = 10000,
    r_min_clamp: float = 1e-6
) -> EmissionContext:
    """
    Chunked version of compute_emission_context for large problems.

    Computes geometric factors in chunks to avoid OOM on GPU.

    Args:
        surface_points: (N_s, 3) on CPU
        observer_points: (N_o, 3) on CPU
        normals: (N_s, 3) on CPU
        device: compute device (GPU)
        chunk_size: number of surface points per chunk
        r_min_clamp: minimum distance clamp

    Returns:
        EmissionContext with tensors on specified device
    """
    N_s = surface_points.shape[0]
    N_o = observer_points.shape[0]

    # Allocate output on device
    r_vec = torch.zeros(N_s, N_o, 3, device=device)
    r_mag = torch.zeros(N_s, N_o, device=device)
    cos_theta = torch.zeros(N_s, N_o, device=device)

    # Move observers to device (typically smaller than surface)
    obs_dev = observer_points.to(device)

    for s0 in range(0, N_s, chunk_size):
        s1 = min(s0 + chunk_size, N_s)

        # Transfer chunk to device
        surf_chunk = surface_points[s0:s1].to(device)
        norm_chunk = normals[s0:s1].to(device)

        # Compute for this chunk
        chunk_r_vec = obs_dev.unsqueeze(0) - surf_chunk.unsqueeze(1)  # (chunk, N_o, 3)
        chunk_r_mag = torch.linalg.norm(chunk_r_vec, dim=-1)
        chunk_r_mag = torch.clamp(chunk_r_mag, min=r_min_clamp)

        chunk_r_hat = chunk_r_vec / chunk_r_mag.unsqueeze(-1)
        chunk_cos = (norm_chunk.unsqueeze(1) * chunk_r_hat).sum(dim=-1)

        # Store results
        r_vec[s0:s1] = chunk_r_hat
        r_mag[s0:s1] = chunk_r_mag
        cos_theta[s0:s1] = chunk_cos

        # Clear cache
        if device.type == 'mps':
            torch.mps.empty_cache()

    r_inv = 1.0 / r_mag
    r_inv_sq = r_inv * r_inv

    return EmissionContext(
        r_vec=r_vec,
        r_mag=r_mag,
        r_inv=r_inv,
        r_inv_sq=r_inv_sq,
        cos_theta=cos_theta
    )


def compute_fwh_kernel(
    Q_dot: Tensor,
    L: Tensor,
    L_dot: Tensor,
    context: EmissionContext,
    c0: float
) -> Tensor:
    """
    Compute the FW-H integrand at a single source timestep.

    Implements the Farassat 1A formulation for a stationary surface:

        kernel = Q̇/r + (L̇·r̂)/(c₀r) + (L·r̂)/r²

    Where:
        - Thickness (monopole): Q̇/r
        - Loading far-field:    (L̇·r̂)/(c₀r)
        - Loading near-field:   (L·r̂)/r²

    Args:
        Q_dot: (N_s,) mass flux time derivative at this timestep
        L: (N_s, 3) loading vector at this timestep
        L_dot: (N_s, 3) loading vector time derivative at this timestep
        context: EmissionContext with pre-computed geometry
        c0: speed of sound

    Returns:
        kernel: (N_s, N_o) integrand values
    """
    # Thickness term: Q̇ / r
    # Q_dot: (N_s,) -> (N_s, 1)
    thickness = Q_dot.unsqueeze(1) * context.r_inv  # (N_s, N_o)

    # Loading far-field term: (L̇ · r̂) / (c₀ · r)
    # L_dot: (N_s, 3) -> (N_s, 1, 3)
    # r_vec (r̂): (N_s, N_o, 3)
    L_dot_r = (L_dot.unsqueeze(1) * context.r_vec).sum(dim=-1)  # (N_s, N_o)
    loading_far = L_dot_r * context.r_inv / c0  # (N_s, N_o)

    # Loading near-field term: (L · r̂) / r²
    # L: (N_s, 3) -> (N_s, 1, 3)
    L_r = (L.unsqueeze(1) * context.r_vec).sum(dim=-1)  # (N_s, N_o)
    loading_near = L_r * context.r_inv_sq  # (N_s, N_o)

    # Total kernel
    kernel = thickness + loading_far + loading_near  # (N_s, N_o)

    return kernel


def accumulate_to_observer_times(
    kernel: Tensor,
    weights: Tensor,
    source_time: float,
    r_mag: Tensor,
    c0: float,
    observer_times: Tensor,
    signal: Tensor,
    dt: float
) -> None:
    """
    Accumulate kernel contributions into observer time signal using linear interpolation.

    For each surface-observer pair, computes the arrival time:
        t_arrival = source_time + r / c0

    Then distributes the contribution to adjacent observer time bins
    using linear interpolation.

    Args:
        kernel: (N_s, N_o) integrand values at this source time
        weights: (N_s,) surface integration weights (area elements)
        source_time: emission time τ
        r_mag: (N_s, N_o) source-observer distances
        c0: speed of sound
        observer_times: (N_t_obs,) observer time grid
        signal: (N_o, N_t_obs) output signal to accumulate into (modified in-place)
        dt: timestep (used for time bin indexing only)

    Note:
        This function modifies `signal` in-place for efficiency.
        The FW-H surface integral yields pressure directly; no dt scaling is applied.
    """
    N_s, N_o = kernel.shape
    N_t_obs = observer_times.shape[0]
    device = kernel.device

    # Compute arrival times: t = tau + r/c0
    t_arrival = source_time + r_mag / c0  # (N_s, N_o)

    # Convert to fractional index in observer time array
    t_start = observer_times[0].item()
    t_idx_frac = (t_arrival - t_start) / dt  # (N_s, N_o)

    # Integer indices for adjacent bins
    t_idx_lo = t_idx_frac.floor().long()  # (N_s, N_o)
    t_idx_hi = t_idx_lo + 1

    # Interpolation weights (linear)
    w_hi = t_idx_frac - t_idx_lo.float()  # weight for upper bin
    w_lo = 1.0 - w_hi  # weight for lower bin

    # Clamp indices to valid range
    valid_lo = (t_idx_lo >= 0) & (t_idx_lo < N_t_obs)
    valid_hi = (t_idx_hi >= 0) & (t_idx_hi < N_t_obs)

    # Weighted kernel: kernel * surface_weight
    # weights: (N_s,) -> (N_s, 1)
    weighted_kernel = kernel * weights.unsqueeze(1)  # (N_s, N_o)

    # Accumulate using scatter_add
    # Need to reshape for scatter: flatten (N_s, N_o) and create observer dimension indices

    # Observer indices for each (s, o) pair
    obs_idx = torch.arange(N_o, device=device).unsqueeze(0).expand(N_s, -1)  # (N_s, N_o)

    # Flatten everything
    obs_flat = obs_idx.reshape(-1)  # (N_s * N_o,)
    t_lo_flat = t_idx_lo.reshape(-1)  # (N_s * N_o,)
    t_hi_flat = t_idx_hi.reshape(-1)  # (N_s * N_o,)
    w_lo_flat = w_lo.reshape(-1)  # (N_s * N_o,)
    w_hi_flat = w_hi.reshape(-1)  # (N_s * N_o,)
    valid_lo_flat = valid_lo.reshape(-1)  # (N_s * N_o,)
    valid_hi_flat = valid_hi.reshape(-1)  # (N_s * N_o,)
    wk_flat = weighted_kernel.reshape(-1)  # (N_s * N_o,)

    # Accumulate lower bin contributions (no dt multiplication - FWH gives pressure directly)
    if valid_lo_flat.any():
        contrib_lo = wk_flat * w_lo_flat
        valid_indices_lo = valid_lo_flat.nonzero(as_tuple=True)[0]

        obs_valid = obs_flat[valid_indices_lo]
        t_valid = t_lo_flat[valid_indices_lo]
        contrib_valid = contrib_lo[valid_indices_lo]

        # Create linear indices into signal[obs, t]
        linear_idx = obs_valid * N_t_obs + t_valid
        signal.view(-1).scatter_add_(0, linear_idx, contrib_valid)

    # Accumulate upper bin contributions
    if valid_hi_flat.any():
        contrib_hi = wk_flat * w_hi_flat
        valid_indices_hi = valid_hi_flat.nonzero(as_tuple=True)[0]

        obs_valid = obs_flat[valid_indices_hi]
        t_valid = t_hi_flat[valid_indices_hi]
        contrib_valid = contrib_hi[valid_indices_hi]

        linear_idx = obs_valid * N_t_obs + t_valid
        signal.view(-1).scatter_add_(0, linear_idx, contrib_valid)


def emission_loop(
    source_terms: FWHSourceTerms,
    context: EmissionContext,
    weights: Tensor,
    source_times: Tensor,
    observer_times: Tensor,
    c0: float,
    dt: float,
    progress_callback: Optional[callable] = None
) -> Tensor:
    """
    Main emission loop over all source timesteps.

    Accumulates FW-H contributions from each source timestep into
    the observer time signal using the Farassat 1A formulation.

    Args:
        source_terms: FWHSourceTerms from compute_source_terms
        context: EmissionContext with pre-computed geometry
        weights: (N_s,) surface integration weights
        source_times: (N_t,) source emission times
        observer_times: (N_t_obs,) observer time grid
        c0: speed of sound
        dt: timestep
        progress_callback: Optional function called with (t_idx, N_t) for progress

    Returns:
        signal: (N_o, N_t_obs) acoustic pressure signal at each observer
    """
    N_t = source_times.shape[0]
    N_o = context.r_mag.shape[1]
    N_t_obs = observer_times.shape[0]
    device = source_terms.device

    # Initialize output signal
    signal = torch.zeros(N_o, N_t_obs, device=device)

    # Ensure weights are on device
    weights = weights.to(device)

    # Loop over source times
    for t_idx in range(N_t):
        # Extract source quantities at this timestep
        Q_dot = source_terms.Q_dot[:, t_idx]      # (N_s,)
        L = source_terms.L[:, t_idx, :]           # (N_s, 3)
        L_dot = source_terms.L_dot[:, t_idx, :]   # (N_s, 3)

        # Compute kernel at this source time
        kernel = compute_fwh_kernel(
            Q_dot=Q_dot,
            L=L,
            L_dot=L_dot,
            context=context,
            c0=c0
        )

        # Accumulate into observer times
        accumulate_to_observer_times(
            kernel=kernel,
            weights=weights,
            source_time=source_times[t_idx].item(),
            r_mag=context.r_mag,
            c0=c0,
            observer_times=observer_times,
            signal=signal,
            dt=dt
        )

        if progress_callback is not None:
            progress_callback(t_idx, N_t)

    return signal
