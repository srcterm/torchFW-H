"""Base classes and data structures for CFD data loading."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor


@dataclass
class CFDSnapshot:
    """Flow data at a single timestep.

    Attributes:
        points: (N, 3) tensor of cell centers or node positions
        fields: Dict mapping field names to (N,) tensors
        time: Physical time of this snapshot
    """
    points: Tensor
    fields: dict[str, Tensor]
    time: float

    def __post_init__(self):
        n_points = self.points.shape[0]
        for name, field in self.fields.items():
            if field.shape[0] != n_points:
                raise ValueError(
                    f"Field '{name}' has {field.shape[0]} values but "
                    f"expected {n_points} (number of points)"
                )

    def to(self, device: torch.device) -> 'CFDSnapshot':
        """Move snapshot to specified device."""
        return CFDSnapshot(
            points=self.points.to(device),
            fields={k: v.to(device) for k, v in self.fields.items()},
            time=self.time
        )

    @property
    def n_points(self) -> int:
        """Number of CFD points."""
        return self.points.shape[0]

    @property
    def bounds(self) -> Tensor:
        """Bounding box as (2, 3) tensor: [[min_x, min_y, min_z], [max_x, max_y, max_z]]."""
        return torch.stack([self.points.min(dim=0).values, self.points.max(dim=0).values])


@dataclass
class CFDMetadata:
    """Time-invariant information about the CFD dataset.

    Attributes:
        times: (N_t,) tensor of all timestep values
        dt: Timestep interval (may be approximate if non-uniform)
        field_names: List of available field names
        bounds: (2, 3) tensor bounding box [[min], [max]]
        n_points: Number of CFD points per snapshot
        uniform_dt: Whether timesteps are uniformly spaced
    """
    times: Tensor
    dt: float
    field_names: list[str]
    bounds: Tensor
    n_points: int
    uniform_dt: bool = True

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps."""
        return self.times.shape[0]

    @property
    def duration(self) -> float:
        """Total time span of the dataset."""
        return (self.times[-1] - self.times[0]).item()


class CFDLoader(ABC):
    """Abstract base class for CFD data loaders.

    Loaders must implement:
    - __iter__: Yield CFDSnapshot objects in time order
    - metadata: Return CFDMetadata for the dataset
    """

    @abstractmethod
    def __iter__(self) -> Iterator[CFDSnapshot]:
        """Yield snapshots in time order."""
        ...

    @property
    @abstractmethod
    def metadata(self) -> CFDMetadata:
        """Get time-invariant metadata about the dataset."""
        ...

    def __len__(self) -> int:
        """Number of timesteps."""
        return self.metadata.n_timesteps

    def validate(self) -> None:
        """Validate dataset properties.

        Raises:
            ValueError: If validation fails (e.g., non-uniform timesteps when required)
        """
        meta = self.metadata

        # Check for required fields
        required = {'rho', 'u', 'v', 'w', 'p'}
        available = set(meta.field_names)
        missing = required - available
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Check timestep uniformity
        if meta.n_timesteps > 1:
            times = meta.times
            dts = torch.diff(times)
            mean_dt = dts.mean()
            if not torch.allclose(dts, mean_dt, rtol=1e-2):
                print(f"Warning: Non-uniform timesteps detected. "
                      f"dt ranges from {dts.min().item():.6f} to {dts.max().item():.6f}")

    def get_snapshot(self, index: int) -> CFDSnapshot:
        """Get snapshot by index.

        Default implementation iterates through; subclasses may optimize.
        """
        for i, snapshot in enumerate(self):
            if i == index:
                return snapshot
        raise IndexError(f"Index {index} out of range")

    def validate_uniform_dt(self, rtol: float = 1e-3) -> float:
        """
        Validate that timesteps are uniform and return dt.

        FW-H solver requires uniform timesteps for FFT-based spectral
        differentiation. This method checks uniformity and raises an
        error if timesteps vary beyond the tolerance.

        Args:
            rtol: Relative tolerance for timestep uniformity check

        Returns:
            dt: The uniform timestep value in seconds

        Raises:
            ValueError: If fewer than 2 timesteps or non-uniform spacing
        """
        times = self.metadata.times

        if len(times) < 2:
            raise ValueError("Need at least 2 timesteps for FW-H solver")

        dt = times[1] - times[0]
        diffs = torch.diff(times)

        if not torch.allclose(diffs, dt, rtol=rtol):
            max_dev = (diffs - dt).abs().max().item()
            mean_dt = diffs.mean().item()
            raise ValueError(
                f"Non-uniform timesteps detected (max deviation: {max_dev:.2e}s, "
                f"mean dt: {mean_dt:.6e}s).\n"
                f"FW-H solver requires uniform dt for spectral differentiation.\n"
                f"Consider resampling your CFD data to uniform intervals."
            )

        return dt.item()
