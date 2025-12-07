"""XDMF/HDF5 loader for CFD data."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch import Tensor

from .base import CFDLoader, CFDMetadata, CFDSnapshot


class XDMFLoader(CFDLoader):
    """Load CFD data from XDMF files with HDF5 backend.

    Handles rectilinear grids (VXVYVZ geometry type) and converts
    node coordinates to cell centers.

    Args:
        path: Path to directory containing .xdmf files, or single .xdmf file
        field_mapping: Optional dict mapping XDMF field names to standard names.
                       Default handles common conventions.
    """

    # Default field name mapping
    DEFAULT_MAPPING = {
        'density': 'rho',
        'pressure': 'p',
        # velocity handled specially (vector -> u, v, w)
    }

    def __init__(
        self,
        path: str | Path,
        field_mapping: dict[str, str] | None = None
    ):
        self.path = Path(path)
        self.field_mapping = {**self.DEFAULT_MAPPING, **(field_mapping or {})}

        # Find all XDMF files
        if self.path.is_file():
            self._xdmf_files = [self.path]
            self._base_dir = self.path.parent
        else:
            self._xdmf_files = sorted(self.path.glob('*.xdmf'))
            self._base_dir = self.path

        if not self._xdmf_files:
            raise FileNotFoundError(f"No .xdmf files found in {self.path}")

        # Parse metadata from files
        self._parse_metadata()

    def _parse_metadata(self) -> None:
        """Parse XDMF files to extract metadata."""
        times = []
        field_names = set()

        for xdmf_path in self._xdmf_files:
            tree = ET.parse(xdmf_path)
            root = tree.getroot()

            # Find Time element
            time_elem = root.find('.//Time')
            if time_elem is not None:
                times.append(float(time_elem.get('Value')))

            # Find Attribute elements (fields)
            for attr in root.findall('.//Attribute'):
                name = attr.get('Name')
                if name:
                    field_names.add(name)

        # Sort by time
        sorted_indices = np.argsort(times)
        self._xdmf_files = [self._xdmf_files[i] for i in sorted_indices]
        times = [times[i] for i in sorted_indices]

        self._times = torch.tensor(times, dtype=torch.float32)

        # Compute dt (may be non-uniform)
        if len(times) > 1:
            dts = np.diff(times)
            self._dt = float(np.mean(dts))
            self._uniform_dt = np.allclose(dts, self._dt, rtol=1e-2)
        else:
            self._dt = 0.0
            self._uniform_dt = True

        # Map field names
        self._raw_field_names = list(field_names)
        self._field_names = []
        for name in field_names:
            if name == 'velocity':
                self._field_names.extend(['u', 'v', 'w'])
            elif name in self.field_mapping:
                self._field_names.append(self.field_mapping[name])
            else:
                self._field_names.append(name)

        # Get grid info from first file
        self._grid_shape, self._bounds, self._n_points = self._parse_grid_info(
            self._xdmf_files[0]
        )

    def _parse_grid_info(self, xdmf_path: Path) -> tuple[tuple, Tensor, int]:
        """Parse grid dimensions and bounds from XDMF file."""
        tree = ET.parse(xdmf_path)
        root = tree.getroot()

        # Find Topology element for dimensions
        topo = root.find('.//Topology')
        dims_str = topo.get('Dimensions')
        node_dims = tuple(int(d) for d in dims_str.split())  # (Nx, Ny, Nz)

        # Cell dimensions are one less than node dimensions
        cell_dims = tuple(d - 1 for d in node_dims)
        n_cells = np.prod(cell_dims)

        # Read grid coordinates to get bounds
        geom = root.find('.//Geometry')
        data_items = geom.findall('DataItem')

        coords = []
        for item in data_items:
            h5_ref = item.text.strip()
            h5_file, h5_path = h5_ref.split(':')
            h5_full_path = self._base_dir / h5_file

            with h5py.File(h5_full_path, 'r') as f:
                coords.append(np.array(f[h5_path]))

        # Bounds from coordinate arrays
        bounds = torch.tensor([
            [coords[0].min(), coords[1].min(), coords[2].min()],
            [coords[0].max(), coords[1].max(), coords[2].max()]
        ], dtype=torch.float32)

        return cell_dims, bounds, int(n_cells)

    def _load_snapshot(self, xdmf_path: Path) -> CFDSnapshot:
        """Load a single snapshot from XDMF/HDF5 files."""
        tree = ET.parse(xdmf_path)
        root = tree.getroot()

        # Get time
        time_elem = root.find('.//Time')
        time = float(time_elem.get('Value'))

        # Load grid coordinates and compute cell centers
        geom = root.find('.//Geometry')
        data_items = geom.findall('DataItem')

        coords = []
        for item in data_items:
            h5_ref = item.text.strip()
            h5_file, h5_path = h5_ref.split(':')
            h5_full_path = self._base_dir / h5_file

            with h5py.File(h5_full_path, 'r') as f:
                coords.append(np.array(f[h5_path], dtype=np.float32))

        # Compute cell centers from node coordinates
        # For rectilinear grid: cell_center = (node[i] + node[i+1]) / 2
        cell_x = 0.5 * (coords[0][:-1] + coords[0][1:])
        cell_y = 0.5 * (coords[1][:-1] + coords[1][1:])
        cell_z = 0.5 * (coords[2][:-1] + coords[2][1:])

        # Create meshgrid of cell centers
        # Note: XDMF uses Fortran ordering (z varies fastest in memory)
        zz, yy, xx = np.meshgrid(cell_z, cell_y, cell_x, indexing='ij')
        points = torch.tensor(
            np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1),
            dtype=torch.float32
        )

        # Load fields
        fields = {}
        for attr in root.findall('.//Attribute'):
            name = attr.get('Name')
            data_item = attr.find('DataItem')
            h5_ref = data_item.text.strip()
            h5_file, h5_path = h5_ref.split(':')
            h5_full_path = self._base_dir / h5_file

            with h5py.File(h5_full_path, 'r') as f:
                data = np.array(f[h5_path], dtype=np.float32)

            if name == 'velocity':
                # Vector field: (Nz, Ny, Nx, 3) -> u, v, w each (N,)
                # Transpose to match point ordering
                data = data.transpose(2, 1, 0, 3)  # -> (Nx, Ny, Nz, 3)
                fields['u'] = torch.tensor(data[..., 0].ravel(), dtype=torch.float32)
                fields['v'] = torch.tensor(data[..., 1].ravel(), dtype=torch.float32)
                fields['w'] = torch.tensor(data[..., 2].ravel(), dtype=torch.float32)
            else:
                # Scalar field: (Nz, Ny, Nx) -> (N,)
                data = data.transpose(2, 1, 0)  # -> (Nx, Ny, Nz)
                mapped_name = self.field_mapping.get(name, name)
                fields[mapped_name] = torch.tensor(data.ravel(), dtype=torch.float32)

        return CFDSnapshot(points=points, fields=fields, time=time)

    @property
    def metadata(self) -> CFDMetadata:
        """Get dataset metadata."""
        return CFDMetadata(
            times=self._times,
            dt=self._dt,
            field_names=self._field_names,
            bounds=self._bounds,
            n_points=self._n_points,
            uniform_dt=self._uniform_dt
        )

    def __iter__(self) -> Iterator[CFDSnapshot]:
        """Yield snapshots in time order."""
        for xdmf_path in self._xdmf_files:
            yield self._load_snapshot(xdmf_path)

    def get_snapshot(self, index: int) -> CFDSnapshot:
        """Get snapshot by index (optimized - doesn't iterate)."""
        if index < 0 or index >= len(self._xdmf_files):
            raise IndexError(f"Index {index} out of range [0, {len(self._xdmf_files)})")
        return self._load_snapshot(self._xdmf_files[index])
