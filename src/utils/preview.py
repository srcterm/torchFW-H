"""Preview utility for FW-H solver setup visualization and diagnostics."""

from typing import Optional
import torch

from ..surfaces.parametric import PermeableSurface, cylinder, sphere, box
from ..surfaces.observers import ObserverArray
from ..loaders.base import CFDLoader
from ..solver.interpolation import ScatteredInterpolator
from ..postprocessing.plots import plot_setup, plot_setup_slices
from .config import FWHConfig
from .device import get_device
from .memory import get_available_memory
import matplotlib.pyplot as plt


def create_surface(config: FWHConfig) -> PermeableSurface:
    """Create permeable surface from configuration."""
    surf_cfg = config.surface

    if surf_cfg.type == 'cylinder':
        return cylinder(
            radius=surf_cfg.radius,
            length=surf_cfg.length,
            center=surf_cfg.center,
            n_theta=surf_cfg.n_theta,
            n_z=surf_cfg.n_z,
            axis=surf_cfg.axis,
            caps=surf_cfg.caps,
            n_cap_radial=surf_cfg.n_cap_radial
        )
    elif surf_cfg.type == 'sphere':
        return sphere(
            radius=surf_cfg.radius,
            center=surf_cfg.center,
            n_theta=surf_cfg.n_theta,
            n_phi=surf_cfg.n_phi
        )
    elif surf_cfg.type == 'box':
        return box(
            extents=surf_cfg.extents,
            center=surf_cfg.center,
            n_per_side=surf_cfg.n_per_side
        )
    else:
        raise ValueError(f"Unknown surface type: {surf_cfg.type}")


def create_observers(config: FWHConfig) -> ObserverArray:
    """Create observer array from configuration."""
    obs_cfg = config.observers

    if obs_cfg.type == 'arc':
        return ObserverArray.arc(
            radius=obs_cfg.radius,
            n=obs_cfg.n,
            plane=obs_cfg.plane,
            center=obs_cfg.center,
            theta_range=obs_cfg.theta_range
        )
    elif obs_cfg.type == 'sphere':
        return ObserverArray.sphere(
            radius=obs_cfg.radius,
            n_theta=obs_cfg.n_theta,
            n_phi=obs_cfg.n_phi,
            center=obs_cfg.center
        )
    elif obs_cfg.type == 'line':
        return ObserverArray.line(
            start=obs_cfg.start,
            end=obs_cfg.end,
            n=obs_cfg.n
        )
    elif obs_cfg.type == 'file':
        return ObserverArray.from_file(obs_cfg.file_path)
    else:
        raise ValueError(f"Unknown observer type: {obs_cfg.type}")


def create_loader(config: FWHConfig) -> CFDLoader:
    """Create CFD data loader from configuration."""
    data_cfg = config.data

    if data_cfg.format == 'xdmf':
        from ..loaders.xdmf import XDMFLoader
        return XDMFLoader(
            path=data_cfg.path,
            field_mapping=data_cfg.field_mapping or None
        )
    else:
        raise ValueError(f"Unknown data format: {data_cfg.format}")


def suggest_f_max(surface: PermeableSurface, c0: float) -> float:
    """
    Suggest cutoff frequency based on surface resolution.

    Uses 6 points per wavelength criterion:
    f_max = c0 / (6 * mean_spacing)

    Args:
        surface: PermeableSurface
        c0: Speed of sound

    Returns:
        Suggested maximum frequency (Hz)
    """
    spacing = surface.mean_spacing
    if spacing <= 0:
        return float('inf')
    return c0 / (6 * spacing)


def preview(
    config: FWHConfig,
    loader: Optional[CFDLoader] = None,
    interactive: bool = True,
    save_path: Optional[str] = None
) -> dict:
    """
    Visualize FW-H setup and print diagnostics.

    Args:
        config: FWHConfig specifying the case
        loader: Optional pre-created CFDLoader (created from config if None)
        interactive: Whether to show interactive visualization
        save_path: Optional path to save screenshot

    Returns:
        Dict with computed parameters and diagnostics
    """
    # Create surface and observers
    surface = create_surface(config)
    observers = create_observers(config)

    # Create or use provided loader
    if loader is None:
        loader = create_loader(config)

    meta = loader.metadata
    c0 = config.case.c0

    # Compute derived parameters
    f_max_suggested = suggest_f_max(surface, c0)
    f_nyquist = 0.5 / meta.dt if meta.dt > 0 else float('inf')

    # Observer time range estimate
    r = torch.cdist(surface.points, observers.positions)
    r_min, r_max = r.min().item(), r.max().item()
    t_obs_min = meta.times[0].item() + r_min / c0
    t_obs_max = meta.times[-1].item() + r_max / c0
    n_obs_samples = int((t_obs_max - t_obs_min) / meta.dt) + 1 if meta.dt > 0 else 0

    # Memory estimates
    N_s = surface.n_points
    N_t = meta.n_timesteps
    N_o = observers.n_observers

    surface_data_gb = (5 * N_s * N_t * 4) / 1e9  # 5 fields, float32
    output_data_gb = (N_o * n_obs_samples * 4) / 1e9

    # Print summary
    print("=" * 60)
    print("FW-H Setup Summary")
    print("=" * 60)
    print()
    print(f"Surface ({config.surface.type}):")
    print(f"  Points:       {N_s:,}")
    print(f"  Total area:   {surface.total_area:.4f} m²")
    print(f"  Mean spacing: {surface.mean_spacing:.6f} m")
    print(f"  Bounds:       {surface.bounds[0].tolist()} to {surface.bounds[1].tolist()}")
    print()
    print(f"Observers ({config.observers.type}):")
    print(f"  Count:        {N_o}")
    if config.observers.type in ('arc', 'sphere'):
        print(f"  Radius:       {config.observers.radius:.1f} m")
    print()
    print(f"CFD Data ({config.data.format}):")
    print(f"  Points:       {meta.n_points:,}")
    print(f"  Timesteps:    {N_t}")
    print(f"  dt:           {meta.dt:.6e} s", end="")
    if not meta.uniform_dt:
        print(" (non-uniform!)")
    else:
        print()
    print(f"  Time range:   [{meta.times[0].item():.6f}, {meta.times[-1].item():.6f}] s")
    print(f"  Fields:       {meta.field_names}")
    print()
    print("Derived Parameters:")
    print(f"  f_max (suggested): {f_max_suggested:.1f} Hz")
    print(f"  f_nyquist:         {f_nyquist:.1f} Hz")
    print(f"  Observer t range:  [{t_obs_min:.6f}, {t_obs_max:.6f}] s ({n_obs_samples} samples)")
    print()
    print("Memory Estimates:")
    print(f"  Surface time series: {surface_data_gb:.4f} GB")
    print(f"  Output signals:      {output_data_gb:.4f} GB")
    print("=" * 60)

    # 3D Visualization
    if config.preview.show_3d and (interactive or save_path):
        print("\nGenerating 3D visualization...", end=" ", flush=True)
      
        plot_setup(
            surface=surface,
            observers=observers,
            cfd_bounds=meta.bounds,
            interactive=interactive,
            save_path=save_path
        )
        print("Done.")

    # 2D Slice plots
    if config.preview.show_slices:
        print("\nGenerating 2D slice plots...", end=" ", flush=True)
        try:
            # Load first snapshot for slice plots
            snapshot = loader.get_snapshot(0)

            # Get slice fields from config
            slice_fields = config.preview.slice_fields

            # Generate slice save path if main save_path is set
            slice_save_path = None
            if config.preview.save_path:
                slice_save_path = config.preview.save_path

            fig = plot_setup_slices(
                snapshot=snapshot,
                surface=surface,
                observers=observers,
                fields=slice_fields,
                save_path=slice_save_path
            )
            print("Done.")

            if interactive:
                plt.show()

        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()

    return {
        'surface': surface,
        'observers': observers,
        'loader': loader,
        'f_max_suggested': f_max_suggested,
        'f_nyquist': f_nyquist,
        'observer_time_range': (t_obs_min, t_obs_max),
        'n_observer_samples': n_obs_samples,
        'memory_gb': {
            'surface_data': surface_data_gb,
            'output': output_data_gb
        }
    }


def test_interpolation(
    config: FWHConfig,
    loader: Optional[CFDLoader] = None,
    field_name: str = 'p',
    device: Optional[torch.device] = None
) -> dict:
    """
    Test interpolation from CFD to surface and report diagnostics.

    Args:
        config: FWHConfig
        loader: Optional pre-created loader
        field_name: Field to interpolate for testing
        device: Target device (auto-detected if None)

    Returns:
        Dict with interpolation diagnostics
    """
    # Resolve device
    if device is None:
        device = get_device(config.solver.device)

    print(f"\nUsing device: {device}")
    available_mem = get_available_memory(device) / (1024**3)
    print(f"Available memory: {available_mem:.2f} GB")

    surface = create_surface(config)
    surface = surface.to(device)

    if loader is None:
        loader = create_loader(config)

    # Get first snapshot and move to device
    snapshot = loader.get_snapshot(0)
    snapshot = snapshot.to(device)

    # Build interpolator (now auto-detects device from inputs)
    interp = ScatteredInterpolator.build(
        source_points=snapshot.points,
        target_points=surface.points,
        k=config.interpolation.k,
        length_scale=config.interpolation.length_scale,
        device=device
    )

    # Interpolate field
    if field_name in snapshot.fields:
        field = snapshot.fields[field_name]
        interpolated = interp(field)

        print(f"\nInterpolation Test ({field_name}):")
        print(f"  Device:         {device}")
        print(f"  Source points:  {snapshot.n_points:,}")
        print(f"  Target points:  {surface.n_points:,}")
        print(f"  k neighbors:    {interp.k}")
        print(f"  Length scale:   {interp.length_scale:.6f}")
        print(f"  Source range:   [{field.min().item():.4f}, {field.max().item():.4f}]")
        print(f"  Interp range:   [{interpolated.min().item():.4f}, {interpolated.max().item():.4f}]")

        diag = interp.diagnostics()
        print(f"  Max weight (mean): {diag['max_weight_mean']:.4f}")
        print(f"  Weight entropy:    {diag['weight_entropy_mean']:.4f}")

        return {
            'interpolator': interp,
            'source_field': field,
            'interpolated_field': interpolated,
            'diagnostics': diag,
            'device': device
        }
    else:
        print(f"Warning: Field '{field_name}' not found. Available: {list(snapshot.fields.keys())}")
        return {'interpolator': interp, 'device': device}


# Required fields for FW-H solver
FWH_FIELDS = ['rho', 'u', 'v', 'w', 'p']


def test_interpolation_all(
    config: FWHConfig,
    loader: Optional[CFDLoader] = None,
    save_stats: Optional[str] = None,
    device: Optional[torch.device] = None
) -> dict:
    """
    Test interpolation of all FW-H fields across ALL timesteps.

    Args:
        config: FWHConfig
        loader: Optional pre-created loader
        save_stats: Optional path to save statistics CSV
        device: Target device (auto-detected if None)

    Returns:
        Dict with timing and field statistics per timestep
    """
    import time
    import numpy as np

    # Resolve device
    if device is None:
        device = get_device(config.solver.device)

    print(f"\nUsing device: {device}")
    available_mem = get_available_memory(device) / (1024**3)
    print(f"Available memory: {available_mem:.2f} GB")

    surface = create_surface(config)
    surface = surface.to(device)

    if loader is None:
        loader = create_loader(config)

    meta = loader.metadata
    n_steps = meta.n_timesteps

    # Check which FW-H fields are available
    snapshot0 = loader.get_snapshot(0)
    snapshot0 = snapshot0.to(device)

    available_fields = [f for f in FWH_FIELDS if f in snapshot0.fields]
    missing_fields = [f for f in FWH_FIELDS if f not in snapshot0.fields]

    if missing_fields:
        print(f"Warning: Missing FW-H fields: {missing_fields}")
    if not available_fields:
        print("Error: No FW-H fields found!")
        return {}

    print(f"\nBuilding interpolator from {snapshot0.n_points:,} source points...")
    t_build_start = time.perf_counter()

    # Build interpolator ONCE (source points are fixed)
    interp = ScatteredInterpolator.build(
        source_points=snapshot0.points,
        target_points=surface.points,
        k=config.interpolation.k,
        length_scale=config.interpolation.length_scale,
        device=device
    )

    t_build = time.perf_counter() - t_build_start
    print(f"Interpolator built in {t_build:.2f} s\n")

    # Initialize statistics tracking
    stats = {
        'timestep': [],
        'time': [],
        'interp_time': [],
    }
    for field in available_fields:
        stats[f'{field}_min'] = []
        stats[f'{field}_max'] = []
        stats[f'{field}_mean'] = []

    print(f"Testing interpolation across {n_steps} timesteps ({len(available_fields)} fields: {', '.join(available_fields)})...")

    # Loop through all timesteps
    for i in range(n_steps):
        t_step_start = time.perf_counter()

        snapshot = loader.get_snapshot(i)
        snapshot = snapshot.to(device)

        # Interpolate all available FW-H fields
        field_stats = []
        for field_name in available_fields:
            field = snapshot.fields[field_name]
            interp_field = interp(field)

            stats[f'{field_name}_min'].append(interp_field.min().item())
            stats[f'{field_name}_max'].append(interp_field.max().item())
            stats[f'{field_name}_mean'].append(interp_field.mean().item())

            field_stats.append(f"{field_name}:[{interp_field.min().item():.2f},{interp_field.max().item():.2f}]")

        t_step = time.perf_counter() - t_step_start

        stats['timestep'].append(i)
        stats['time'].append(snapshot.time)
        stats['interp_time'].append(t_step)

        # Progress output
        print(f"[{i+1:3d}/{n_steps}] t={snapshot.time:.6f} | {t_step:.3f}s | {' '.join(field_stats)}")

    # Summary statistics
    interp_times = np.array(stats['interp_time'])
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Device:         {device}")
    print(f"Timesteps:      {n_steps}")
    print(f"Fields:         {', '.join(available_fields)}")
    print(f"Build time:     {t_build:.2f} s")
    print(f"Total interp:   {interp_times.sum():.2f} s")
    print(f"Mean per step:  {interp_times.mean():.3f} s")
    print(f"Std:            {interp_times.std():.3f} s")
    print()
    print("Field Statistics (interpolated, across all timesteps):")
    for field_name in available_fields:
        f_min = min(stats[f'{field_name}_min'])
        f_max = max(stats[f'{field_name}_max'])
        f_mean = np.mean(stats[f'{field_name}_mean'])
        print(f"  {field_name:4s}: min={f_min:10.4f}, max={f_max:10.4f}, mean={f_mean:10.4f}")

    # Optionally save to CSV
    if save_stats:
        import csv
        with open(save_stats, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            header = ['timestep', 'time', 'interp_time']
            for field in available_fields:
                header.extend([f'{field}_min', f'{field}_max', f'{field}_mean'])
            writer.writerow(header)
            # Data rows
            for i in range(n_steps):
                row = [stats['timestep'][i], stats['time'][i], stats['interp_time'][i]]
                for field in available_fields:
                    row.extend([stats[f'{field}_min'][i], stats[f'{field}_max'][i], stats[f'{field}_mean'][i]])
                writer.writerow(row)
        print(f"\nStatistics saved to: {save_stats}")

    return {
        'interpolator': interp,
        'stats': stats,
        'available_fields': available_fields,
        'build_time': t_build,
        'device': device
    }
