"""Scattered-data interpolation for CFD to surface mapping."""

from dataclasses import dataclass

import torch
from torch import Tensor


def auto_length_scale(distances: Tensor) -> float:
    """
    Estimate Gaussian kernel length scale from k-NN distances.

    Uses 2× the mean nearest-neighbor distance as a reasonable default.

    Args:
        distances: (N, k) k-NN distances

    Returns:
        Suggested length scale
    """
    return distances[:, 0].mean().item() * 2.0


def acoustic_length_scale(c0: float, f_max: float) -> float:
    """
    Compute length scale for spatial anti-aliasing.

    Uses lambda_min / 4 criterion (4 points per wavelength).

    Args:
        c0: Speed of sound
        f_max: Maximum frequency of interest

    Returns:
        Length scale for Gaussian kernel
    """
    return c0 / (4 * f_max)


@dataclass
class ScatteredInterpolator:
    """k-NN interpolation with Gaussian kernel weights.

    Interpolates field values from source points (e.g., CFD cell centers)
    to target points (e.g., FW-H surface points).

    Attributes:
        indices: (N_target, k) indices into source points
        weights: (N_target, k) normalized interpolation weights
        k: Number of neighbors used
        length_scale: Gaussian kernel length scale
    """
    indices: Tensor
    weights: Tensor
    k: int
    length_scale: float

    @classmethod
    def build(
        cls,
        source_points: Tensor,
        target_points: Tensor,
        k: int = 8,
        length_scale: float | str = 'auto',
        target_chunk_size: int = 512,
        source_chunk_size: int = 100000
    ) -> 'ScatteredInterpolator':
        """
        Build interpolator from source to target points.

        Uses k-NN with Gaussian radial basis function weighting.
        Processes in chunks to handle large point sets (millions of points).

        Args:
            source_points: (N_source, 3) CFD points
            target_points: (N_target, 3) surface points
            k: Number of nearest neighbors
            length_scale: Gaussian kernel width. 'auto' estimates from data.
            target_chunk_size: Process this many target points at a time
            source_chunk_size: Process this many source points at a time

        Returns:
            ScatteredInterpolator ready for use
        """
        N_source = source_points.shape[0]
        N_target = target_points.shape[0]
        k = min(k, N_source)

        # For each target point, track best k neighbors across all source chunks
        all_indices = torch.zeros(N_target, k, dtype=torch.long)
        all_distances = torch.full((N_target, k), float('inf'))

        # Process target points in chunks
        for t0 in range(0, N_target, target_chunk_size):
            t1 = min(t0 + target_chunk_size, N_target)
            target_chunk = target_points[t0:t1]
            n_chunk = t1 - t0

            # Track best k for this target chunk
            chunk_distances = torch.full((n_chunk, k), float('inf'))
            chunk_indices = torch.zeros(n_chunk, k, dtype=torch.long)

            # Process source points in chunks
            for s0 in range(0, N_source, source_chunk_size):
                s1 = min(s0 + source_chunk_size, N_source)
                source_chunk = source_points[s0:s1]

                # Compute distances: (n_chunk, source_chunk_size)
                dist = torch.cdist(target_chunk, source_chunk)

                # Find k best in this source chunk
                k_local = min(k, dist.shape[1])
                local_dist, local_idx = torch.topk(dist, k_local, dim=1, largest=False)

                # Convert local indices to global
                local_idx = local_idx + s0

                # Merge with running best k
                # Concatenate current best with new candidates
                merged_dist = torch.cat([chunk_distances, local_dist], dim=1)
                merged_idx = torch.cat([chunk_indices, local_idx], dim=1)

                # Sort and keep best k
                sorted_dist, sort_order = torch.sort(merged_dist, dim=1)
                chunk_distances = sorted_dist[:, :k]
                chunk_indices = torch.gather(merged_idx, 1, sort_order[:, :k])

            # Store results for this target chunk
            all_distances[t0:t1] = chunk_distances
            all_indices[t0:t1] = chunk_indices
            print('Chunked k-NN:', t1, 'of', N_target, 'target points processed.')

        # Determine length scale
        if length_scale == 'auto':
            length_scale = auto_length_scale(all_distances)

        # Compute Gaussian weights: exp(-d² / (2 * sigma²))
        weights = torch.exp(-all_distances**2 / (2 * length_scale**2))

        # Normalize weights to sum to 1 for each target point
        weights = weights / weights.sum(dim=1, keepdim=True)

        return cls(
            indices=all_indices,
            weights=weights,
            k=k,
            length_scale=length_scale
        )

    def __call__(self, field: Tensor) -> Tensor:
        """
        Interpolate a field from source to target points.

        Args:
            field: (N_source,) field values at source points

        Returns:
            (N_target,) interpolated values at target points
        """
        # Gather neighbor values: (N_target, k)
        neighbor_values = field[self.indices]
        # Weighted sum
        return (neighbor_values * self.weights).sum(dim=-1)

    def interpolate_vector(self, field: Tensor) -> Tensor:
        """
        Interpolate a vector field from source to target points.

        Args:
            field: (N_source, D) vector field values

        Returns:
            (N_target, D) interpolated vector values
        """
        D = field.shape[1]
        # Gather neighbor values: (N_target, k, D)
        neighbor_values = field[self.indices]
        # Weighted sum along k dimension
        return (neighbor_values * self.weights.unsqueeze(-1)).sum(dim=1)

    def to(self, device: torch.device) -> 'ScatteredInterpolator':
        """Move interpolator to specified device."""
        return ScatteredInterpolator(
            indices=self.indices.to(device),
            weights=self.weights.to(device),
            k=self.k,
            length_scale=self.length_scale
        )

    @property
    def n_target(self) -> int:
        """Number of target points."""
        return self.indices.shape[0]

    def diagnostics(self) -> dict:
        """Get interpolation diagnostics.

        Returns:
            Dict with interpolation quality metrics
        """
        # Weight statistics
        max_weight = self.weights.max(dim=1).values
        weight_entropy = -(self.weights * torch.log(self.weights + 1e-10)).sum(dim=1)

        return {
            'k': self.k,
            'length_scale': self.length_scale,
            'max_weight_mean': max_weight.mean().item(),
            'max_weight_min': max_weight.min().item(),
            'weight_entropy_mean': weight_entropy.mean().item(),
        }
