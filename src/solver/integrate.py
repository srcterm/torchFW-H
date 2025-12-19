"""
Main FW-H solver integration.

Orchestrates the complete FW-H acoustic analogy computation:
1. Validate inputs and compute parameters
2. Build interpolator (ScatteredInterpolator)
3. Interpolate CFD fields to FW-H surface
4. Compute source terms (Q, L, derivatives)
5. Setup observer time grid
6. Emission loop over source times
7. Apply normalization and return results

The solver implements the porous (permeable) FW-H formulation
for a stationary integration surface.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import json

import torch
from torch import Tensor
import h5py

from ..surfaces.parametric import PermeableSurface
from ..surfaces.observers import ObserverArray
from ..loaders.base import CFDLoader
from ..utils.config import FWHConfig
from .interpolation import ScatteredInterpolator
from .source_terms import compute_source_terms, FWHSourceTerms
from .derivatives import suggest_f_max
from .emission import (
    compute_observer_time_grid,
    compute_emission_context,
    compute_emission_context_chunked,
    emission_loop
)


def _get_device(device_spec: Optional[str]) -> torch.device:
    """Resolve device specification."""
    if device_spec is None or device_spec == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    return torch.device(device_spec)


@dataclass
class FWHResult:
    """
    Results from FW-H acoustic computation.

    Attributes:
        observer_times: (N_t_obs,) time array for observer signals
        pressure: (N_o, N_t_obs) acoustic pressure at each observer
        observer_positions: (N_o, 3) observer positions
        observer_labels: list of observer identifiers
        config: configuration used for the computation
        metadata: additional metadata from the computation
    """
    observer_times: Tensor          # (N_t_obs,)
    pressure: Tensor                # (N_o, N_t_obs)
    observer_positions: Tensor      # (N_o, 3)
    observer_labels: list[str]
    config: FWHConfig
    metadata: dict = field(default_factory=dict)

    @property
    def n_observers(self) -> int:
        """Number of observers."""
        return self.pressure.shape[0]

    @property
    def n_timesteps(self) -> int:
        """Number of observer timesteps."""
        return self.pressure.shape[1]

    @property
    def dt(self) -> float:
        """Observer timestep."""
        if self.n_timesteps < 2:
            return 0.0
        return (self.observer_times[1] - self.observer_times[0]).item()

    @property
    def duration(self) -> float:
        """Total signal duration."""
        return (self.observer_times[-1] - self.observer_times[0]).item()

    def to(self, device: torch.device) -> 'FWHResult':
        """Move result tensors to specified device."""
        return FWHResult(
            observer_times=self.observer_times.to(device),
            pressure=self.pressure.to(device),
            observer_positions=self.observer_positions.to(device),
            observer_labels=self.observer_labels.copy(),
            config=self.config,
            metadata=self.metadata.copy()
        )

    def to_hdf5(self, path: str | Path) -> None:
        """
        Save results to HDF5 file.

        File structure:
            /observer_times  (N_t_obs,)
            /pressure        (N_o, N_t_obs)
            /observer_positions (N_o, 3)
            /observer_labels (N_o,) string array
            /metadata        group with attrs

        Args:
            path: output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, 'w') as f:
            # Arrays
            f.create_dataset('observer_times', data=self.observer_times.cpu().numpy())
            f.create_dataset('pressure', data=self.pressure.cpu().numpy())
            f.create_dataset('observer_positions', data=self.observer_positions.cpu().numpy())

            # String labels
            dt = h5py.special_dtype(vlen=str)
            labels_ds = f.create_dataset('observer_labels', (len(self.observer_labels),), dtype=dt)
            for i, label in enumerate(self.observer_labels):
                labels_ds[i] = label

            # Metadata
            meta_grp = f.create_group('metadata')
            for key, value in self.metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    meta_grp.attrs[key] = value
                elif isinstance(value, (list, tuple)):
                    meta_grp.attrs[key] = str(value)
                else:
                    meta_grp.attrs[key] = str(value)

            # Store config as JSON string
            from dataclasses import asdict
            config_dict = asdict(self.config)
            f.attrs['config'] = json.dumps(config_dict, default=str)

    @classmethod
    def from_hdf5(cls, path: str | Path) -> 'FWHResult':
        """
        Load results from HDF5 file.

        Args:
            path: input file path

        Returns:
            FWHResult loaded from file
        """
        from ..utils.config import FWHConfig

        path = Path(path)

        with h5py.File(path, 'r') as f:
            observer_times = torch.from_numpy(f['observer_times'][:])
            pressure = torch.from_numpy(f['pressure'][:])
            observer_positions = torch.from_numpy(f['observer_positions'][:])
            observer_labels = [s.decode() if isinstance(s, bytes) else s
                               for s in f['observer_labels'][:]]

            # Load metadata
            metadata = {}
            if 'metadata' in f:
                for key in f['metadata'].attrs:
                    metadata[key] = f['metadata'].attrs[key]

            # Load config (stored as JSON)
            config_json = f.attrs.get('config', '{}')
            # Note: full config reconstruction would require load_config
            # For now, just create default config
            config = FWHConfig()

        return cls(
            observer_times=observer_times,
            pressure=pressure,
            observer_positions=observer_positions,
            observer_labels=observer_labels,
            config=config,
            metadata=metadata
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'observer_times': self.observer_times.cpu().tolist(),
            'pressure': self.pressure.cpu().tolist(),
            'observer_positions': self.observer_positions.cpu().tolist(),
            'observer_labels': self.observer_labels,
            'metadata': self.metadata
        }


def fwh_solve(
    surface: PermeableSurface,
    observers: ObserverArray,
    loader: CFDLoader,
    config: FWHConfig,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    verbose: bool = True
) -> FWHResult:
    """
    Run the complete FW-H acoustic analogy solver.

    Steps:
    1. Validate inputs and determine parameters (f_max, device)
    2. Build k-NN interpolator from CFD points to surface
    3. Load and interpolate all CFD timesteps to surface
    4. Compute source terms (Q, L) and their time derivatives
    5. Setup observer time grid based on distances
    6. Run emission loop, accumulating contributions
    7. Apply 1/(4π) normalization and return

    Args:
        surface: PermeableSurface defining the integration surface
        observers: ObserverArray with observer positions
        loader: CFDLoader providing the CFD data
        config: FWHConfig with solver parameters
        progress_callback: Optional callback(stage, current, total) for progress
        verbose: print progress messages

    Returns:
        FWHResult with observer times and pressure signals

    Raises:
        ValueError: if inputs are invalid or CFD data missing required fields
    """
    # Resolve device
    device = _get_device(config.solver.device)
    if verbose:
        print(f"FW-H Solver starting on {device}")

    # Validate CFD data
    loader.validate()
    dt = loader.validate_uniform_dt()

    meta = loader.metadata
    if verbose:
        print(f"  CFD data: {meta.n_points:,} points, {meta.n_timesteps} timesteps, dt={dt:.6e}s")

    # Determine f_max
    if config.solver.f_max == 'auto':
        f_max = suggest_f_max(surface.mean_spacing, config.case.c0)
        if verbose:
            print(f"  Auto f_max: {f_max:.1f} Hz (from mean spacing {surface.mean_spacing:.4f}m)")
    else:
        f_max = float(config.solver.f_max)
        if verbose:
            print(f"  f_max: {f_max:.1f} Hz (from config)")

    # Progress helper
    def report(stage: str, current: int, total: int):
        if progress_callback:
            progress_callback(stage, current, total)
        if verbose:
            print(f"  [{stage}] {current}/{total}", end='\r' if current < total else '\n')

    # === Step 1: Build interpolator ===
    report("Interpolator", 0, 1)

    # Get first snapshot for points
    first_snapshot = next(iter(loader))
    source_points = first_snapshot.points

    # Build interpolator
    interp = ScatteredInterpolator.build(
        source_points=source_points,
        target_points=surface.points,
        k=config.interpolation.k,
        length_scale=config.interpolation.length_scale,
        device=device,
        verbose=verbose
    )
    report("Interpolator", 1, 1)

    # === Step 2: Interpolate all timesteps ===
    N_s = surface.n_points
    N_t = meta.n_timesteps

    # Pre-allocate surface field arrays
    rho_surf = torch.zeros(N_s, N_t, device=device)
    u_surf = torch.zeros(N_s, N_t, device=device)
    v_surf = torch.zeros(N_s, N_t, device=device)
    w_surf = torch.zeros(N_s, N_t, device=device)
    p_surf = torch.zeros(N_s, N_t, device=device)

    source_times = meta.times.clone()

    if verbose:
        print(f"  Interpolating {N_t} timesteps to {N_s:,} surface points...")

    for t_idx, snapshot in enumerate(loader):
        # Interpolate each field
        rho_surf[:, t_idx] = interp(snapshot.fields['rho'])
        u_surf[:, t_idx] = interp(snapshot.fields['u'])
        v_surf[:, t_idx] = interp(snapshot.fields['v'])
        w_surf[:, t_idx] = interp(snapshot.fields['w'])
        p_surf[:, t_idx] = interp(snapshot.fields['p'])

        report("Interpolation", t_idx + 1, N_t)

    # === Step 3: Compute source terms ===
    report("Source terms", 0, 1)

    # Move normals to device
    normals = surface.normals.to(device)

    source_terms = compute_source_terms(
        rho=rho_surf,
        u=u_surf,
        v=v_surf,
        w=w_surf,
        p=p_surf,
        normals=normals,
        dt=dt,
        c0=config.case.c0,
        rho0=config.case.rho0,
        p0=config.case.p0,
        f_max=f_max
    )

    # Free interpolated fields to save memory
    del rho_surf, u_surf, v_surf, w_surf, p_surf
    if device.type == 'mps':
        torch.mps.empty_cache()

    report("Source terms", 1, 1)

    if verbose:
        max_mach = source_terms.M_sq.max().sqrt().item()
        print(f"  Max Mach number: {max_mach:.3f}")

    # === Step 4: Setup observer time grid ===
    report("Time grid", 0, 1)

    observer_times, obs_dt, t_offset = compute_observer_time_grid(
        source_times=source_times,
        surface_points=surface.points,
        observer_points=observers.positions,
        c0=config.case.c0
    )
    observer_times = observer_times.to(device)

    if verbose:
        print(f"  Observer time grid: {len(observer_times)} steps, "
              f"range [{observer_times[0].item():.6f}, {observer_times[-1].item():.6f}]s")

    report("Time grid", 1, 1)

    # === Step 5: Compute emission context ===
    report("Geometry", 0, 1)

    # Use chunked version for large problems
    if N_s * observers.n_observers > 10_000_000:
        context = compute_emission_context_chunked(
            surface_points=surface.points,
            observer_points=observers.positions,
            normals=surface.normals,
            device=device,
            chunk_size=10000
        )
    else:
        # Move to device and compute directly
        surf_pts = surface.points.to(device)
        obs_pts = observers.positions.to(device)
        norms = surface.normals.to(device)

        context = compute_emission_context(
            surface_points=surf_pts,
            observer_points=obs_pts,
            normals=norms
        )

    report("Geometry", 1, 1)

    # === Step 6: Emission loop ===
    if verbose:
        print(f"  Running emission loop ({N_t} source times)...")

    def emission_progress(t_idx: int, total: int):
        report("Emission", t_idx + 1, total)

    signal = emission_loop(
        source_terms=source_terms,
        context=context,
        weights=surface.weights,
        source_times=source_times,
        observer_times=observer_times,
        c0=config.case.c0,
        dt=dt,
        progress_callback=emission_progress
    )

    # === Step 7: Apply 1/(4π) normalization ===
    pressure = signal / (4.0 * torch.pi)

    # Build result
    metadata = {
        'f_max': f_max,
        'dt_source': dt,
        'dt_observer': obs_dt,
        'n_surface_points': N_s,
        'n_timesteps_source': N_t,
        'n_timesteps_observer': len(observer_times),
        'c0': config.case.c0,
        'rho0': config.case.rho0,
        'device': str(device),
    }

    result = FWHResult(
        observer_times=observer_times.cpu(),
        pressure=pressure.cpu(),
        observer_positions=observers.positions,
        observer_labels=observers.labels,
        config=config,
        metadata=metadata
    )

    if verbose:
        p_rms = (pressure ** 2).mean(dim=1).sqrt()
        print(f"  RMS pressure range: [{p_rms.min().item():.2e}, {p_rms.max().item():.2e}] Pa")
        print("FW-H Solver complete.")

    return result
