"""Parametric surface generation for FW-H integration."""

from dataclasses import dataclass
from typing import Tuple
import math

import torch
from torch import Tensor
import numpy as np


def _estimate_knn_subsample_size(n_points: int) -> int:
    """
    Compute subsample size for mean spacing estimation.

    Scales as sqrt(N) with bounds: min 1000, max 10000.
    Reimplemented here to avoid circular import.
    """
    n_sample = int(math.sqrt(n_points) * 10)
    return min(n_points, max(1000, min(n_sample, 10000)))


def _get_available_memory(device: torch.device) -> float:
    """Get available memory on device. Reimplemented to avoid circular import."""
    if device.type == 'cuda':
        free_mem, _ = torch.cuda.mem_get_info(device)
        return free_mem * 0.7
    elif device.type == 'mps':
        try:
            import psutil
            return psutil.virtual_memory().available * 0.5
        except ImportError:
            return 4 * (1024**3)
    else:
        try:
            import psutil
            return psutil.virtual_memory().available * 0.6
        except ImportError:
            return 8 * (1024**3)


def _compute_simple_chunks(n: int, device: torch.device) -> int:
    """Compute chunk size for n×n distance matrix."""
    available = _get_available_memory(device) * 0.5  # Use 50% for spacing estimate
    bytes_per_pair = 4 * 1.5  # float32 + overhead
    chunk = int(math.sqrt(available / bytes_per_pair))
    return max(256, min(chunk, n))


def _estimate_mean_spacing(points: Tensor, seed: int = 42) -> float:
    """
    Estimate mean nearest-neighbor spacing via subsampled k-NN.

    Uses random subsampling with fixed seed for reproducibility.
    Scales as O(sqrt(N)) instead of O(N²).

    Args:
        points: (N, 3) point positions
        seed: Random seed for reproducibility

    Returns:
        Estimated mean nearest-neighbor distance
    """
    N = points.shape[0]
    if N < 2:
        return 0.0

    # Determine subsample size (scales as sqrt(N))
    n_sample = _estimate_knn_subsample_size(N)

    # Subsample with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    if n_sample < N:
        indices = torch.randperm(N, generator=generator)[:n_sample]
        sample = points[indices]
    else:
        sample = points

    n = sample.shape[0]
    device = sample.device

    # Compute chunk size for this subsample
    chunk_size = _compute_simple_chunks(n, device)

    # Find k=2 nearest neighbors (self + nearest) via chunked cdist
    # We need k=2 because k=1 would just return self (distance=0)
    k = 2
    all_distances = torch.full((n, k), float('inf'), device=device)

    for t0 in range(0, n, chunk_size):
        t1 = min(t0 + chunk_size, n)
        target_chunk_pts = sample[t0:t1]
        n_chunk = t1 - t0

        chunk_distances = torch.full((n_chunk, k), float('inf'), device=device)

        for s0 in range(0, n, chunk_size):
            s1 = min(s0 + chunk_size, n)
            source_chunk_pts = sample[s0:s1]

            dist = torch.cdist(target_chunk_pts, source_chunk_pts)

            # Find k best in this source chunk
            k_local = min(k, dist.shape[1])
            local_dist, _ = torch.topk(dist, k_local, dim=1, largest=False)

            # Merge with running best k
            merged_dist = torch.cat([chunk_distances, local_dist], dim=1)
            sorted_dist, _ = torch.sort(merged_dist, dim=1)
            chunk_distances = sorted_dist[:, :k]

        all_distances[t0:t1] = chunk_distances

    # Return mean of nearest neighbor distance (column 1, since column 0 is self=0)
    # Actually for self-distance, if point is in both source and target, distance=0
    # So we take the second-smallest (index 1) which is the true nearest neighbor
    return all_distances[:, 1].mean().item()


@dataclass
class PermeableSurface:
    """Point-cloud representation of a permeable FW-H surface.

    Attributes:
        points: (N_s, 3) surface point positions
        normals: (N_s, 3) unit outward normal vectors
        weights: (N_s,) area weights for integration (Jacobian × dA)
    """
    points: Tensor
    normals: Tensor
    weights: Tensor

    def __post_init__(self):
        """Validate tensor shapes."""
        n = self.points.shape[0]
        if self.normals.shape[0] != n:
            raise ValueError(f"normals has {self.normals.shape[0]} rows, expected {n}")
        if self.weights.shape[0] != n:
            raise ValueError(f"weights has {self.weights.shape[0]} elements, expected {n}")

        # Ensure normals are unit vectors
        norms = torch.linalg.norm(self.normals, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
            # Normalize if not already
            self.normals = self.normals / norms.unsqueeze(1)

    def to(self, device: torch.device) -> 'PermeableSurface':
        """Move surface to specified device."""
        return PermeableSurface(
            points=self.points.to(device),
            normals=self.normals.to(device),
            weights=self.weights.to(device)
        )

    @property
    def n_points(self) -> int:
        """Number of surface points."""
        return self.points.shape[0]

    @property
    def total_area(self) -> float:
        """Total surface area (sum of weights)."""
        return self.weights.sum().item()

    @property
    def mean_spacing(self) -> float:
        """
        Mean nearest-neighbor distance between surface points.

        Uses subsampled k-NN estimation for O(sqrt(N)) scaling
        instead of O(N²) full distance matrix.
        """
        return _estimate_mean_spacing(self.points)

    @property
    def bounds(self) -> Tensor:
        """Bounding box as (2, 3) tensor: [[min], [max]]."""
        return torch.stack([
            self.points.min(dim=0).values,
            self.points.max(dim=0).values
        ])

    @property
    def centroid(self) -> Tensor:
        """Centroid of surface points."""
        return self.points.mean(dim=0)


def cylinder(
    radius: float,
    length: float,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_theta: int = 64,
    n_z: int = 32,
    axis: str = 'z',
    caps: bool = True,
    n_cap_radial: int | None = None
) -> PermeableSurface:
    """Generate a cylindrical permeable surface.

    The cylinder is centered at `center` with its axis along the specified direction.

    Args:
        radius: Cylinder radius
        length: Cylinder length (total, centered at `center`)
        center: (x, y, z) center position
        n_theta: Number of circumferential points
        n_z: Number of axial points on lateral surface
        axis: Axis direction ('x', 'y', or 'z')
        caps: Whether to include end caps
        n_cap_radial: Number of radial points on caps (default: n_theta // 4)

    Returns:
        PermeableSurface with points, normals, and area weights
    """
    center = torch.tensor(center, dtype=torch.float32)

    # Angular spacing
    theta = torch.linspace(0, 2 * np.pi, n_theta + 1)[:-1]
    dtheta = 2 * np.pi / n_theta

    # Axial spacing
    z_local = torch.linspace(-length / 2, length / 2, n_z)
    dz = length / (n_z - 1) if n_z > 1 else length

    # Lateral surface
    theta_grid, z_grid = torch.meshgrid(theta, z_local, indexing='ij')
    theta_flat = theta_grid.reshape(-1)
    z_flat = z_grid.reshape(-1)

    # Local coordinates (cylinder axis along z_local)
    x_local = radius * torch.cos(theta_flat)
    y_local = radius * torch.sin(theta_flat)

    # Normals point radially outward
    nx_local = torch.cos(theta_flat)
    ny_local = torch.sin(theta_flat)
    nz_local = torch.zeros_like(theta_flat)

    # Area weight: R * dtheta * dz (Jacobian for cylindrical coords)
    lateral_weights = torch.full_like(theta_flat, radius * dtheta * dz)

    # Stack local coordinates
    points_local = torch.stack([x_local, y_local, z_flat], dim=1)
    normals_local = torch.stack([nx_local, ny_local, nz_local], dim=1)
    weights = lateral_weights

    # Add end caps if requested
    if caps:
        n_cap_radial = n_cap_radial or max(n_theta // 4, 4)

        for z_cap, nz_sign in [(-length / 2, -1.0), (length / 2, 1.0)]:
            # Radial points (excluding r=0 to avoid singularity)
            r_cap = torch.linspace(radius / n_cap_radial, radius, n_cap_radial)
            dr = radius / n_cap_radial

            theta_cap, r_cap_grid = torch.meshgrid(theta, r_cap, indexing='ij')
            theta_cap_flat = theta_cap.reshape(-1)
            r_cap_flat = r_cap_grid.reshape(-1)

            x_cap = r_cap_flat * torch.cos(theta_cap_flat)
            y_cap = r_cap_flat * torch.sin(theta_cap_flat)
            z_cap_pts = torch.full_like(x_cap, z_cap)

            cap_points = torch.stack([x_cap, y_cap, z_cap_pts], dim=1)

            # Cap normals: +z or -z
            cap_normals = torch.zeros_like(cap_points)
            cap_normals[:, 2] = nz_sign

            # Area weight: r * dr * dtheta (Jacobian for polar coords)
            cap_weights = r_cap_flat * dr * dtheta

            points_local = torch.cat([points_local, cap_points], dim=0)
            normals_local = torch.cat([normals_local, cap_normals], dim=0)
            weights = torch.cat([weights, cap_weights], dim=0)

    # Rotate to desired axis orientation
    if axis.lower() == 'z':
        points = points_local
        normals = normals_local
    elif axis.lower() == 'x':
        # Rotate: z->x, x->y, y->z
        points = points_local[:, [2, 0, 1]]
        normals = normals_local[:, [2, 0, 1]]
    elif axis.lower() == 'y':
        # Rotate: z->y, x->z, y->x
        points = points_local[:, [1, 2, 0]]
        normals = normals_local[:, [1, 2, 0]]
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    # Translate to center
    points = points + center

    return PermeableSurface(points=points, normals=normals, weights=weights)


def sphere(
    radius: float,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_theta: int = 64,
    n_phi: int = 32
) -> PermeableSurface:
    """Generate a spherical permeable surface.

    Uses (theta, phi) parameterization where:
    - theta: azimuthal angle [0, 2π)
    - phi: polar angle [0, π] (from +z axis)

    Args:
        radius: Sphere radius
        center: (x, y, z) center position
        n_theta: Number of azimuthal points
        n_phi: Number of polar points

    Returns:
        PermeableSurface with points, normals, and area weights
    """
    center = torch.tensor(center, dtype=torch.float32)

    # Angular grids
    theta = torch.linspace(0, 2 * np.pi, n_theta + 1)[:-1]
    # Avoid poles (phi=0 and phi=pi) where Jacobian is zero
    phi = torch.linspace(np.pi / n_phi, np.pi - np.pi / n_phi, n_phi - 1)

    dtheta = 2 * np.pi / n_theta
    dphi = np.pi / n_phi

    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
    theta_flat = theta_grid.reshape(-1)
    phi_flat = phi_grid.reshape(-1)

    # Spherical to Cartesian
    sin_phi = torch.sin(phi_flat)
    cos_phi = torch.cos(phi_flat)
    sin_theta = torch.sin(theta_flat)
    cos_theta = torch.cos(theta_flat)

    x = radius * sin_phi * cos_theta
    y = radius * sin_phi * sin_theta
    z = radius * cos_phi

    points = torch.stack([x, y, z], dim=1) + center

    # Normals: radially outward (same as position unit vector for sphere)
    normals = torch.stack([
        sin_phi * cos_theta,
        sin_phi * sin_theta,
        cos_phi
    ], dim=1)

    # Area weight: R² sin(phi) * dtheta * dphi (Jacobian for spherical coords)
    weights = radius**2 * sin_phi * dtheta * dphi

    return PermeableSurface(points=points, normals=normals, weights=weights)


def box(
    extents: Tuple[float, float, float],
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_per_side: int | Tuple[int, int, int] = 16
) -> PermeableSurface:
    """Generate a box (rectangular prism) permeable surface.

    Args:
        extents: (Lx, Ly, Lz) full dimensions of the box
        center: (x, y, z) center position
        n_per_side: Points per unit length, or (nx, ny, nz) for each dimension

    Returns:
        PermeableSurface with points, normals, and area weights
    """
    center = torch.tensor(center, dtype=torch.float32)
    extents = torch.tensor(extents, dtype=torch.float32)
    half = extents / 2

    if isinstance(n_per_side, int):
        # Scale by side length
        n_per_side = (
            max(int(n_per_side * extents[0].item()), 2),
            max(int(n_per_side * extents[1].item()), 2),
            max(int(n_per_side * extents[2].item()), 2)
        )

    nx, ny, nz = n_per_side

    all_points = []
    all_normals = []
    all_weights = []

    # Face definitions: (fixed_axis, fixed_value_sign, u_axis, v_axis, normal_sign)
    faces = [
        (0, -1, 1, 2, -1),  # -x face
        (0, +1, 1, 2, +1),  # +x face
        (1, -1, 0, 2, -1),  # -y face
        (1, +1, 0, 2, +1),  # +y face
        (2, -1, 0, 1, -1),  # -z face
        (2, +1, 0, 1, +1),  # +z face
    ]

    n_axes = [nx, ny, nz]

    for fixed_axis, fixed_sign, u_axis, v_axis, normal_sign in faces:
        nu = n_axes[u_axis]
        nv = n_axes[v_axis]

        # Grid on this face
        u = torch.linspace(-half[u_axis], half[u_axis], nu)
        v = torch.linspace(-half[v_axis], half[v_axis], nv)
        du = extents[u_axis] / (nu - 1) if nu > 1 else extents[u_axis].item()
        dv = extents[v_axis] / (nv - 1) if nv > 1 else extents[v_axis].item()

        u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
        u_flat = u_grid.reshape(-1)
        v_flat = v_grid.reshape(-1)

        # Build 3D coordinates
        coords = torch.zeros(u_flat.shape[0], 3)
        coords[:, fixed_axis] = fixed_sign * half[fixed_axis]
        coords[:, u_axis] = u_flat
        coords[:, v_axis] = v_flat

        # Normal vector
        normal = torch.zeros(3)
        normal[fixed_axis] = normal_sign
        normals = normal.unsqueeze(0).expand(u_flat.shape[0], -1).clone()

        # Area weight (planar face, Jacobian = 1)
        weights = torch.full((u_flat.shape[0],), du * dv)

        all_points.append(coords)
        all_normals.append(normals)
        all_weights.append(weights)

    points = torch.cat(all_points, dim=0) + center
    normals = torch.cat(all_normals, dim=0)
    weights = torch.cat(all_weights, dim=0)

    return PermeableSurface(points=points, normals=normals, weights=weights)
