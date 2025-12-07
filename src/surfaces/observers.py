"""Observer array definitions for FW-H acoustic predictions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import json
import csv

import torch
from torch import Tensor
import numpy as np


@dataclass
class ObserverArray:
    """Collection of observer (microphone) positions.

    Attributes:
        positions: (N_o, 3) observer positions
        labels: Identifiers for each observer (for output labeling)
    """
    positions: Tensor
    labels: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Generate default labels if not provided."""
        if not self.labels:
            self.labels = [f'obs_{i:03d}' for i in range(self.positions.shape[0])]
        elif len(self.labels) != self.positions.shape[0]:
            raise ValueError(
                f"labels has {len(self.labels)} elements but "
                f"positions has {self.positions.shape[0]} points"
            )

    def to(self, device: torch.device) -> 'ObserverArray':
        """Move observer positions to specified device."""
        return ObserverArray(
            positions=self.positions.to(device),
            labels=self.labels.copy()
        )

    @property
    def n_observers(self) -> int:
        """Number of observers."""
        return self.positions.shape[0]

    @property
    def centroid(self) -> Tensor:
        """Centroid of observer positions."""
        return self.positions.mean(dim=0)

    @classmethod
    def arc(
        cls,
        radius: float,
        n: int,
        plane: str = 'xy',
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        theta_range: Tuple[float, float] = (0.0, 360.0),
        labels: list[str] | None = None
    ) -> 'ObserverArray':
        """Create observers along a circular arc.

        Args:
            radius: Arc radius
            n: Number of observers
            plane: Plane containing the arc ('xy', 'xz', or 'yz')
            center: (x, y, z) center of the arc
            theta_range: (start, end) angles in degrees
            labels: Optional custom labels

        Returns:
            ObserverArray positioned along the arc
        """
        center = torch.tensor(center, dtype=torch.float32)
        theta_start = np.radians(theta_range[0])
        theta_end = np.radians(theta_range[1])

        theta = torch.linspace(theta_start, theta_end, n)

        # Generate points in 2D then map to 3D plane
        u = radius * torch.cos(theta)
        v = radius * torch.sin(theta)
        zeros = torch.zeros_like(u)

        plane = plane.lower()
        if plane == 'xy':
            positions = torch.stack([u, v, zeros], dim=1)
        elif plane == 'xz':
            positions = torch.stack([u, zeros, v], dim=1)
        elif plane == 'yz':
            positions = torch.stack([zeros, u, v], dim=1)
        else:
            raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got '{plane}'")

        positions = positions + center

        # Generate angle-based labels if not provided
        if labels is None:
            angles_deg = torch.linspace(theta_range[0], theta_range[1], n)
            labels = [f'{plane}_{angle:.0f}deg' for angle in angles_deg.tolist()]

        return cls(positions=positions, labels=labels)

    @classmethod
    def sphere(
        cls,
        radius: float,
        n_theta: int,
        n_phi: int,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        labels: list[str] | None = None
    ) -> 'ObserverArray':
        """Create observers distributed on a sphere.

        Uses (theta, phi) parameterization:
        - theta: azimuthal angle [0, 2π)
        - phi: polar angle [0, π] (from +z axis)

        Args:
            radius: Sphere radius
            n_theta: Number of azimuthal positions
            n_phi: Number of polar positions
            center: (x, y, z) center of the sphere
            labels: Optional custom labels

        Returns:
            ObserverArray positioned on the sphere
        """
        center = torch.tensor(center, dtype=torch.float32)

        theta = torch.linspace(0, 2 * np.pi, n_theta + 1)[:-1]
        phi = torch.linspace(np.pi / n_phi, np.pi - np.pi / n_phi, n_phi - 1)

        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
        theta_flat = theta_grid.reshape(-1)
        phi_flat = phi_grid.reshape(-1)

        sin_phi = torch.sin(phi_flat)
        cos_phi = torch.cos(phi_flat)
        sin_theta = torch.sin(theta_flat)
        cos_theta = torch.cos(theta_flat)

        x = radius * sin_phi * cos_theta
        y = radius * sin_phi * sin_theta
        z = radius * cos_phi

        positions = torch.stack([x, y, z], dim=1) + center

        if labels is None:
            labels = [
                f'th{np.degrees(t):.0f}_ph{np.degrees(p):.0f}'
                for t, p in zip(theta_flat.tolist(), phi_flat.tolist())
            ]

        return cls(positions=positions, labels=labels)

    @classmethod
    def line(
        cls,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        n: int,
        labels: list[str] | None = None
    ) -> 'ObserverArray':
        """Create observers along a line.

        Args:
            start: (x, y, z) start position
            end: (x, y, z) end position
            n: Number of observers
            labels: Optional custom labels

        Returns:
            ObserverArray positioned along the line
        """
        start = torch.tensor(start, dtype=torch.float32)
        end = torch.tensor(end, dtype=torch.float32)

        t = torch.linspace(0, 1, n).unsqueeze(1)
        positions = start + t * (end - start)

        if labels is None:
            labels = [f'line_{i:03d}' for i in range(n)]

        return cls(positions=positions, labels=labels)

    @classmethod
    def from_file(cls, path: str | Path) -> 'ObserverArray':
        """Load observers from a file.

        Supported formats:
        - CSV: columns x, y, z (and optional 'label')
        - JSON: list of {x, y, z, label?} objects or {positions: [[x,y,z],...], labels: [...]}

        Args:
            path: Path to the file

        Returns:
            ObserverArray loaded from file
        """
        path = Path(path)

        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                data = json.load(f)

            if isinstance(data, list):
                # List of point objects
                positions = torch.tensor(
                    [[p['x'], p['y'], p['z']] for p in data],
                    dtype=torch.float32
                )
                labels = [p.get('label', f'obs_{i:03d}') for i, p in enumerate(data)]
            else:
                # Dict with positions and optional labels
                positions = torch.tensor(data['positions'], dtype=torch.float32)
                labels = data.get('labels', None)

        elif path.suffix.lower() == '.csv':
            positions = []
            labels = []
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    positions.append([float(row['x']), float(row['y']), float(row['z'])])
                    labels.append(row.get('label', f'obs_{i:03d}'))
            positions = torch.tensor(positions, dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return cls(positions=positions, labels=labels)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'positions': self.positions.tolist(),
            'labels': self.labels
        }

    def save(self, path: str | Path) -> None:
        """Save observers to file (JSON format)."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
