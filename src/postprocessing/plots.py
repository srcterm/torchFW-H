"""Visualization functions for FW-H solver."""

from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor
import pyvista as pv
from matplotlib import pyplot as plt


def _tensor_to_numpy(t: Tensor) -> np.ndarray:
    """Convert tensor to numpy, handling device transfer."""
    return t.detach().cpu().numpy()


def plot_surface(
    surface: 'PermeableSurface',
    field: Optional[Tensor] = None,
    field_name: str = 'Field',
    cmap: str = 'viridis',
    show_normals: bool = False,
    normal_scale: float = 0.1,
    plotter: Optional['pv.Plotter'] = None,
    **kwargs
) -> 'pv.Plotter':
    """
    Plot a permeable surface with optional scalar field.

    Args:
        surface: PermeableSurface to plot
        field: Optional (N_s,) scalar field to color the surface
        field_name: Name for the field (colorbar label)
        cmap: Colormap name
        show_normals: Whether to show normal vectors as arrows
        normal_scale: Scale factor for normal arrows
        plotter: Existing PyVista plotter to add to
        **kwargs: Additional arguments passed to pyvista add_points

    Returns:
        PyVista plotter
    """

    from ..surfaces.parametric import PermeableSurface

    points = _tensor_to_numpy(surface.points)

    if plotter is None:
        plotter = pv.Plotter()

    # Create point cloud
    cloud = pv.PolyData(points)

    if field is not None:
        cloud[field_name] = _tensor_to_numpy(field)
        plotter.add_points(
            cloud,
            scalars=field_name,
            cmap=cmap,
            point_size=5,
            render_points_as_spheres=True,
            **kwargs
        )
    else:
        plotter.add_points(
            cloud,
            color='blue',
            point_size=5,
            render_points_as_spheres=True,
            **kwargs
        )

    if show_normals:
        normals = _tensor_to_numpy(surface.normals)
        # Scale normals by mean spacing
        scale = surface.mean_spacing * normal_scale * 10
        arrows = pv.Arrow()
        glyphs = cloud.glyph(
            orient=normals,
            scale=False,
            factor=scale,
            geom=arrows
        )
        plotter.add_mesh(glyphs, color='red', opacity=0.5)

    return plotter


def plot_observers(
    observers: 'ObserverArray',
    surface: Optional['PermeableSurface'] = None,
    plotter: Optional['pv.Plotter'] = None,
    observer_color: str = 'red',
    observer_size: float = 15,
    show_labels: bool = False,
    **kwargs
) -> 'pv.Plotter':
    """
    Plot observer positions.

    Args:
        observers: ObserverArray to plot
        surface: Optional PermeableSurface to show for context
        plotter: Existing PyVista plotter to add to
        observer_color: Color for observer points
        observer_size: Point size for observers
        show_labels: Whether to show observer labels
        **kwargs: Additional arguments

    Returns:
        PyVista plotter
    """

    from ..surfaces.observers import ObserverArray
    from ..surfaces.parametric import PermeableSurface

    positions = _tensor_to_numpy(observers.positions)

    if plotter is None:
        plotter = pv.Plotter()

    # Plot surface first if provided
    if surface is not None:
        plotter = plot_surface(surface, plotter=plotter, opacity=0.3)

    # Add observer points
    obs_cloud = pv.PolyData(positions)
    plotter.add_points(
        obs_cloud,
        color=observer_color,
        point_size=observer_size,
        render_points_as_spheres=True,
        **kwargs
    )

    if show_labels:
        for i, label in enumerate(observers.labels):
            plotter.add_point_labels(
                [positions[i]],
                [label],
                font_size=10,
                point_size=1
            )

    return plotter


def plot_cfd_slice(
    snapshot: 'CFDSnapshot',
    field_name: str,
    plane: str = 'xy',
    coord: float = 0.0,
    tolerance: float = None,
    cmap: str = 'viridis',
    plotter: Optional['pv.Plotter'] = None,
    vmin: float = None,
    vmax: float = None
) -> 'pv.Plotter':
    """
    Plot a 2D slice of CFD data.

    Args:
        snapshot: CFDSnapshot containing field data
        field_name: Name of field to plot ('rho', 'p', 'u', 'v', 'w')
        plane: Slice plane ('xy', 'xz', or 'yz')
        coord: Coordinate value for the slice (in the normal direction)
        tolerance: Tolerance for selecting points near the slice (auto if None)
        cmap: Colormap name
        plotter: Existing PyVista plotter to add to
        vmin, vmax: Colorbar limits

    Returns:
        PyVista plotter
    """

    import matplotlib.pyplot as plt
    from ..loaders.base import CFDSnapshot

    points = _tensor_to_numpy(snapshot.points)
    field = _tensor_to_numpy(snapshot.fields[field_name])

    plane = plane.lower()
    if plane == 'xy':
        normal_axis = 2
    elif plane == 'xz':
        normal_axis = 1
    elif plane == 'yz':
        normal_axis = 0
    else:
        raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got '{plane}'")

    # Auto tolerance based on grid spacing
    if tolerance is None:
        unique_coords = np.unique(points[:, normal_axis])
        if len(unique_coords) > 1:
            tolerance = np.diff(unique_coords).min() * 0.6
        else:
            tolerance = 0.01

    # Select points near the slice
    mask = np.abs(points[:, normal_axis] - coord) < tolerance
    if not mask.any():
        raise ValueError(
            f"No points found near {plane} plane at "
            f"{'xyz'[normal_axis]} = {coord} (tolerance = {tolerance})"
        )

    slice_points = points[mask]
    values = field[mask]

    if plotter is None:
        plotter = pv.Plotter()

    cloud = pv.PolyData(slice_points)
    cloud[field_name] = values

    clim = [vmin, vmax] if vmin is not None and vmax is not None else None
    plotter.add_points(
        cloud,
        scalars=field_name,
        cmap=cmap,
        point_size=5,
        render_points_as_spheres=True,
        clim=clim
    )
    plotter.add_title(f'{field_name} slice at {"xyz"[normal_axis]} = {coord:.3f}')

    return plotter


def plot_setup(
    surface: 'PermeableSurface',
    observers: 'ObserverArray',
    cfd_bounds: Optional[Tensor] = None,
    interactive: bool = True,
    save_path: Optional[str] = None
) -> 'pv.Plotter':
    """
    Plot complete FW-H setup (surface, observers, CFD domain).

    Args:
        surface: PermeableSurface
        observers: ObserverArray
        cfd_bounds: Optional (2, 3) tensor of CFD bounding box
        interactive: Whether to show interactive plot
        save_path: Optional path to save screenshot

    Returns:
        PyVista plotter
    """
    from ..surfaces.parametric import PermeableSurface
    from ..surfaces.observers import ObserverArray

    plotter = pv.Plotter()

    # Add surface
    surf_pts = _tensor_to_numpy(surface.points)
    surf_cloud = pv.PolyData(surf_pts)
    plotter.add_points(
        surf_cloud,
        color='blue',
        point_size=3,
        render_points_as_spheres=True,
        label='FW-H Surface'
    )

    # Add observers
    obs_pts = _tensor_to_numpy(observers.positions)
    obs_cloud = pv.PolyData(obs_pts)
    plotter.add_points(
        obs_cloud,
        color='red',
        point_size=10,
        render_points_as_spheres=True,
        label='Observers'
    )

    # Add CFD bounding box if provided
    if cfd_bounds is not None:
        bounds_np = _tensor_to_numpy(cfd_bounds)
        box = pv.Box(bounds=[
            bounds_np[0, 0], bounds_np[1, 0],
            bounds_np[0, 1], bounds_np[1, 1],
            bounds_np[0, 2], bounds_np[1, 2]
        ])
        plotter.add_mesh(
            box,
            style='wireframe',
            color='gray',
            line_width=2,
            label='CFD Domain'
        )

    plotter.add_legend()
    plotter.add_axes()

    if save_path:
        plotter.screenshot(save_path)

    if interactive:
        plotter.show()

    return plotter


def plot_setup_slices(
        snapshot: 'CFDSnapshot',
        surface: 'PermeableSurface',
        observers: 'ObserverArray',
        fields: list[str] = None,
        slice_coords: dict = None,
        figsize: Tuple[int, int] = (15, 8),
        save_path: Optional[str] = None,
        show_geometry: bool = None
    ) -> plt.Figure:
        """
        Plot 2D slice views of the FW-H setup showing CFD domain, surface, and observers.

        Creates 3 subplots for XY, XZ, YZ planes with overlaid FW-H surface and
        observer positions. If levelset field exists (e.g., from XDMF/JAXFluids),
        also shows geometry as solid region.

        Args:
            snapshot: CFDSnapshot containing field data
            surface: PermeableSurface to overlay
            observers: ObserverArray to overlay
            fields: Ignored (kept for compatibility)
            slice_coords: Dict with slice coordinates {'xy': z, 'xz': y, 'yz': x}.
                            If None, uses geometry center (if levelset) or domain center.
            figsize: Figure size
            save_path: Optional path to save figure
            show_geometry: Whether to plot geometry from levelset.
                           If None, auto-detects based on 'levelset' field presence.

        Returns:
            Matplotlib figure
        """
        # Auto-detect geometry plotting based on levelset availability
        has_levelset = 'levelset' in snapshot.fields
        if show_geometry is None:
            show_geometry = has_levelset

        points = _tensor_to_numpy(snapshot.points)
        surf_pts = _tensor_to_numpy(surface.points)
        obs_pts = _tensor_to_numpy(observers.positions)

        # Compute domain bounds
        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, z_max = points.max(axis=0)

        print(f"  CFD domain: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]")

        # Find geometry center from levelset=0 (if available), otherwise use domain center
        if has_levelset:
            levelset = _tensor_to_numpy(snapshot.fields['levelset'])
            near_zero = np.abs(levelset) < 0.1
            if near_zero.any():
                geom_pts = points[near_zero]
                geom_center = geom_pts.mean(axis=0)
                print(f"  Geometry center (from levelset): ({geom_center[0]:.2f}, {geom_center[1]:.2f}, {geom_center[2]:.2f})")
            else:
                geom_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
        else:
            # No levelset - use domain center for slices
            geom_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
            print(f"  Using domain center for slices (no levelset field)")

        # Default slice coordinates: geometry center (not domain center)
        if slice_coords is None:
            slice_coords = {
                'xy': geom_center[2],  # z at geometry center
                'xz': geom_center[1],  # y at geometry center
                'yz': geom_center[0]   # x at geometry center
            }
        print(f"  Slice coords: XY@z={slice_coords['xy']:.2f}, XZ@y={slice_coords['xz']:.2f}, YZ@x={slice_coords['yz']:.2f}")

        # Plane definitions: (normal_axis, u_axis, v_axis, xlabel, ylabel)
        planes = {
            'xy': (2, 0, 1, 'X', 'Y'),
            'xz': (1, 0, 2, 'X', 'Z'),
            'yz': (0, 1, 2, 'Y', 'Z')
        }

        n_planes = 3

        fig, axes = plt.subplots(1, n_planes, figsize=figsize)
        axes = axes.flatten()

        # Compute tolerance for slice extraction based on grid spacing
        # Use larger multiplier for better geometry visualization (more points in slice)
        unique_x = np.unique(points[:, 0])
        unique_y = np.unique(points[:, 1])
        unique_z = np.unique(points[:, 2])

        dx = np.diff(unique_x).mean() if len(unique_x) > 1 else 0.1
        dy = np.diff(unique_y).mean() if len(unique_y) > 1 else 0.1
        dz = np.diff(unique_z).mean() if len(unique_z) > 1 else 0.1

        tolerances = {
            'xy': dz * 3.0,   # Larger tolerance for better coverage
            'xz': dy * 3.0,
            'yz': dx * 3.0
        }

        for col, (plane_name, (normal_axis, u_axis, v_axis, xlabel, ylabel)) in enumerate(planes.items()):
            ax = axes[col]
            coord = slice_coords[plane_name]
            tol = tolerances[plane_name]

            # Extract CFD points near slice
            mask = np.abs(points[:, normal_axis] - coord) < tol
            n_points = mask.sum()

            if not mask.any():
                ax.text(0.5, 0.5, f'No data at {plane_name}={coord:.2f}',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            # print(f"  {plane_name.upper()} slice: {n_points} points extracted (tol={tol:.4f})")

            u_cfd = points[mask, u_axis]
            v_cfd = points[mask, v_axis]

            # Get levelset values for this slice (if available)
            levelset_slice = None
            if has_levelset:
                levelset_slice = _tensor_to_numpy(snapshot.fields['levelset'])[mask]

            # Surface points near slice
            surf_tol = tol * 5
            surf_mask = np.abs(surf_pts[:, normal_axis] - coord) < surf_tol
            u_surf = surf_pts[surf_mask, u_axis] if surf_mask.any() else np.array([])
            v_surf = surf_pts[surf_mask, v_axis] if surf_mask.any() else np.array([])

            # All observer positions (projected)
            u_obs = obs_pts[:, u_axis]
            v_obs = obs_pts[:, v_axis]

            # Draw domain outline (rectangle)
            u_min, u_max = u_cfd.min(), u_cfd.max()
            v_min, v_max = v_cfd.min(), v_cfd.max()
            rect = plt.Rectangle((u_min, v_min), u_max - u_min, v_max - v_min,
                                   fill=False, edgecolor='gray', linewidth=1.5, linestyle='-', zorder=1)
            ax.add_patch(rect)

            # Plot geometry (levelset < 0) as solid black - only if levelset available
            if show_geometry and levelset_slice is not None:
                geom_mask = levelset_slice < 0
                if geom_mask.any():
                    ax.scatter(u_cfd[geom_mask], v_cfd[geom_mask], c='black', s=5,
                                alpha=1.0, label='Geometry', zorder=5)

            # FW-H surface
            if len(u_surf) > 0:
                ax.scatter(u_surf, v_surf, c='cyan', s=2, marker='o',
                            edgecolors='blue', linewidths=0.7, label='FW-H Surface', zorder=10)

            # Observers
            ax.scatter(u_obs, v_obs, c='lime', s=5, marker='^',
                        edgecolors='darkgreen', linewidths=1.5, label='Observers', zorder=11)

            ax.set_xlabel(f'{xlabel} (m)', fontsize=12)
            ax.set_ylabel(f'{ylabel} (m)', fontsize=12)
            
            # Set same axis limits for all plots based on full domain + observers
            all_u = np.concatenate([u_cfd, u_obs])
            all_v = np.concatenate([v_cfd, v_obs])
            margin = 0.5
            ax.set_xlim(all_u.min() - margin, all_u.max() + margin)
            ax.set_ylim(all_v.min() - margin, all_v.max() + margin)
            ax.set_aspect('equal')

            axis_name = 'XYZ'[normal_axis]
            ax.set_title(f'{plane_name.upper()} Slice @ {axis_name}={coord:.2f}', fontsize=13, fontweight='bold')

            ax.grid(True, alpha=0.3, linestyle='--')

            if col == 0:
                ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved slice plot to: {save_path}")

        return fig


def plot_psd(
    frequencies: Tensor,
    psd: Tensor,
    labels: Optional[list] = None,
    title: str = "Power Spectral Density",
    p_ref: float = 20e-6,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Power Spectral Density vs frequency.

    Args:
        frequencies: (N_freq,) frequency array in Hz
        psd: (N_freq,) or (N_obs, N_freq) PSD values in Pa^2/Hz
        labels: Optional list of observer labels
        title: Plot title
        p_ref: Reference pressure for dB conversion (default 20 uPa)
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    freqs_np = _tensor_to_numpy(frequencies)
    psd_np = _tensor_to_numpy(psd)

    fig, ax = plt.subplots(figsize=figsize)

    # Handle single vs multiple observers
    if psd_np.ndim == 1:
        psd_np = psd_np[np.newaxis, :]

    n_obs = psd_np.shape[0]

    # Convert to dB/Hz referenced to p_ref^2
    psd_db = 10 * np.log10(np.clip(psd_np, 1e-40, None) / (p_ref ** 2))

    # Use different colors for each observer
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_obs, 10)))

    for i in range(n_obs):
        label = labels[i] if labels else f"Observer {i+1}"
        color = colors[i % len(colors)]
        ax.semilogx(freqs_np, psd_db[i], label=label, color=color)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz re 20 uPa)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if n_obs <= 10:
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PSD plot to: {save_path}")

    return fig


def plot_spl(
    frequencies: Tensor,
    spl: Tensor,
    labels: Optional[list] = None,
    title: str = "SPL Spectrum",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot SPL vs frequency spectrum.

    Args:
        frequencies: (N_freq,) frequency array in Hz
        spl: (N_freq,) or (N_obs, N_freq) SPL values in dB
        labels: Optional list of observer labels
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    freqs_np = _tensor_to_numpy(frequencies)
    spl_np = _tensor_to_numpy(spl)

    fig, ax = plt.subplots(figsize=figsize)

    # Handle single vs multiple observers
    if spl_np.ndim == 1:
        spl_np = spl_np[np.newaxis, :]

    n_obs = spl_np.shape[0]

    # Use different colors for each observer
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_obs, 10)))

    for i in range(n_obs):
        label = labels[i] if labels else f"Observer {i+1}"
        color = colors[i % len(colors)]
        ax.semilogx(freqs_np, spl_np[i], label=label, color=color)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPL (dB re 20 uPa)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if n_obs <= 10:
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved SPL plot to: {save_path}")

    return fig


def plot_directivity(
    observer_positions: Tensor,
    levels: Tensor,
    plane: str = 'xy',
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    title: str = "Directivity Pattern",
    figsize: Tuple[float, float] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot polar directivity pattern from observer OASPL or SPL values.

    Args:
        observer_positions: (N_obs, 3) observer positions
        levels: (N_obs,) OASPL or SPL values in dB for each observer
        plane: Projection plane ('xy', 'xz', or 'yz')
        center: Reference center point for angle computation
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    obs_np = _tensor_to_numpy(observer_positions)
    levels_np = _tensor_to_numpy(levels)
    center_np = np.array(center)

    # Compute relative positions
    rel_pos = obs_np - center_np

    # Get angles based on plane
    plane = plane.lower()
    if plane == 'xy':
        angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
    elif plane == 'xz':
        angles = np.arctan2(rel_pos[:, 2], rel_pos[:, 0])
    elif plane == 'yz':
        angles = np.arctan2(rel_pos[:, 2], rel_pos[:, 1])
    else:
        raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got '{plane}'")

    # Sort by angle for smooth line
    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    levels_sorted = levels_np[sort_idx]

    # Close the loop
    angles_plot = np.concatenate([angles_sorted, [angles_sorted[0]]])
    levels_plot = np.concatenate([levels_sorted, [levels_sorted[0]]])

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

    ax.plot(angles_plot, levels_plot, 'b-', linewidth=2)
    ax.scatter(angles_sorted, levels_sorted, c='blue', s=30, zorder=5)

    # Set radial limits with some padding
    level_min = levels_np.min()
    level_max = levels_np.max()
    level_range = level_max - level_min
    ax.set_rlim(level_min - 0.1 * level_range, level_max + 0.1 * level_range)

    ax.set_title(title, pad=20)
    ax.set_theta_zero_location('E')  # 0 degrees at right (standard)
    ax.set_theta_direction(1)  # Counter-clockwise

    # Add plane label
    ax.text(0.02, 0.98, f'{plane.upper()} plane', transform=ax.transAxes,
            fontsize=10, verticalalignment='top')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved directivity plot to: {save_path}")

    return fig


def compute_source_rms(
    source_terms: 'FWHSourceTerms'
) -> Tuple[Tensor, Tensor]:
    """
    Compute RMS of source terms for surface visualization.

    Args:
        source_terms: FWHSourceTerms from compute_source_terms

    Returns:
        Q_rms: (N_s,) RMS of mass flux Q over time
        L_rms: (N_s,) RMS of loading magnitude |L| over time
    """
    import torch

    # Q RMS: sqrt(mean(Q^2)) over time axis
    Q_rms = torch.sqrt((source_terms.Q ** 2).mean(dim=1))

    # L magnitude at each timestep, then RMS
    L_mag = torch.linalg.norm(source_terms.L, dim=-1)  # (N_s, N_t)
    L_rms = torch.sqrt((L_mag ** 2).mean(dim=1))  # (N_s,)

    return Q_rms, L_rms


def plot_source_distribution(
    surface: 'PermeableSurface',
    source_terms: 'FWHSourceTerms',
    field: str = 'Q',
    title: Optional[str] = None,
    cmap: str = 'hot',
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot source term distribution on the FW-H surface.

    Args:
        surface: PermeableSurface
        source_terms: FWHSourceTerms from compute_source_terms
        field: Which field to plot ('Q' for mass flux, 'L' for loading)
        title: Plot title (auto-generated if None)
        cmap: Colormap
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D

    Q_rms, L_rms = compute_source_rms(source_terms)

    if field.upper() == 'Q':
        values = _tensor_to_numpy(Q_rms)
        field_label = 'Mass Flux RMS |Q|'
        units = 'kg/(m²·s)'
    elif field.upper() == 'L':
        values = _tensor_to_numpy(L_rms)
        field_label = 'Loading RMS |L|'
        units = 'Pa'
    else:
        raise ValueError(f"field must be 'Q' or 'L', got '{field}'")

    points = _tensor_to_numpy(surface.points)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=values, cmap=cmap, s=10
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(f'{field_label} ({units})')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    if title is None:
        title = f'Source Distribution: {field_label}'
    ax.set_title(title)

    # Equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) / 2
    mid_y = (points[:, 1].max() + points[:, 1].min()) / 2
    mid_z = (points[:, 2].max() + points[:, 2].min()) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved source distribution plot to: {save_path}")

    return fig
