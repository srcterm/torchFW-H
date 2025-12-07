"""Visualization functions for FW-H solver."""

from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor

# Try to import PyVista, fall back to matplotlib-only if not available
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
) -> Optional['pv.Plotter']:
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
        PyVista plotter (if available) or None
    """
    from ..surfaces.parametric import PermeableSurface

    points = _tensor_to_numpy(surface.points)

    if HAS_PYVISTA:
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

    else:
        # Matplotlib fallback
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if field is not None:
            field_np = _tensor_to_numpy(field)
            scatter = ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=field_np, cmap=cmap, s=5
            )
            plt.colorbar(scatter, label=field_name)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Permeable Surface')
        plt.show()
        return None


def plot_observers(
    observers: 'ObserverArray',
    surface: Optional['PermeableSurface'] = None,
    plotter: Optional['pv.Plotter'] = None,
    observer_color: str = 'red',
    observer_size: float = 15,
    show_labels: bool = False,
    **kwargs
) -> Optional['pv.Plotter']:
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
        PyVista plotter (if available) or None
    """
    from ..surfaces.observers import ObserverArray
    from ..surfaces.parametric import PermeableSurface

    positions = _tensor_to_numpy(observers.positions)

    if HAS_PYVISTA:
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

    else:
        # Matplotlib fallback
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if surface is not None:
            surf_pts = _tensor_to_numpy(surface.points)
            ax.scatter(
                surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2],
                s=1, alpha=0.3, label='Surface'
            )

        ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=observer_color, s=observer_size, label='Observers'
        )

        if show_labels:
            for i, label in enumerate(observers.labels):
                ax.text(positions[i, 0], positions[i, 1], positions[i, 2], label)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('Observer Positions')
        plt.show()
        return None


def plot_cfd_slice(
    snapshot: 'CFDSnapshot',
    field_name: str,
    plane: str = 'xy',
    coord: float = 0.0,
    tolerance: float = None,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (10, 8),
    vmin: float = None,
    vmax: float = None
) -> plt.Figure:
    """
    Plot a 2D slice of CFD data.

    Args:
        snapshot: CFDSnapshot containing field data
        field_name: Name of field to plot ('rho', 'p', 'u', 'v', 'w')
        plane: Slice plane ('xy', 'xz', or 'yz')
        coord: Coordinate value for the slice (in the normal direction)
        tolerance: Tolerance for selecting points near the slice (auto if None)
        cmap: Colormap name
        figsize: Figure size
        vmin, vmax: Colorbar limits

    Returns:
        Matplotlib figure
    """
    from ..loaders.base import CFDSnapshot

    points = _tensor_to_numpy(snapshot.points)
    field = _tensor_to_numpy(snapshot.fields[field_name])

    plane = plane.lower()
    if plane == 'xy':
        normal_axis = 2
        u_axis, v_axis = 0, 1
        xlabel, ylabel = 'X', 'Y'
    elif plane == 'xz':
        normal_axis = 1
        u_axis, v_axis = 0, 2
        xlabel, ylabel = 'X', 'Z'
    elif plane == 'yz':
        normal_axis = 0
        u_axis, v_axis = 1, 2
        xlabel, ylabel = 'Y', 'Z'
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

    u = points[mask, u_axis]
    v = points[mask, v_axis]
    values = field[mask]

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(u, v, c=values, cmap=cmap, s=1, vmin=vmin, vmax=vmax)
    plt.colorbar(scatter, label=field_name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
    ax.set_title(f'{field_name} slice at {"xyz"[normal_axis]} = {coord:.3f}')

    return fig


def plot_setup(
    surface: 'PermeableSurface',
    observers: 'ObserverArray',
    cfd_bounds: Optional[Tensor] = None,
    interactive: bool = True,
    save_path: Optional[str] = None
) -> Optional['pv.Plotter']:
    """
    Plot complete FW-H setup (surface, observers, CFD domain).

    Args:
        surface: PermeableSurface
        observers: ObserverArray
        cfd_bounds: Optional (2, 3) tensor of CFD bounding box
        interactive: Whether to show interactive plot
        save_path: Optional path to save screenshot

    Returns:
        PyVista plotter (if available and interactive) or None
    """
    from ..surfaces.parametric import PermeableSurface
    from ..surfaces.observers import ObserverArray

    if HAS_PYVISTA:
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
        else:
            return plotter

    else:
        # Matplotlib fallback
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf_pts = _tensor_to_numpy(surface.points)
        obs_pts = _tensor_to_numpy(observers.positions)

        ax.scatter(
            surf_pts[:, 0], surf_pts[:, 1], surf_pts[:, 2],
            s=1, alpha=0.5, c='blue', label='FW-H Surface'
        )
        ax.scatter(
            obs_pts[:, 0], obs_pts[:, 1], obs_pts[:, 2],
            s=50, c='red', label='Observers'
        )

        if cfd_bounds is not None:
            bounds_np = _tensor_to_numpy(cfd_bounds)
            # Draw wireframe box
            for i in range(2):
                for j in range(2):
                    ax.plot3D(
                        [bounds_np[0, 0], bounds_np[1, 0]],
                        [bounds_np[i, 1], bounds_np[i, 1]],
                        [bounds_np[j, 2], bounds_np[j, 2]],
                        'gray', alpha=0.5
                    )
                    ax.plot3D(
                        [bounds_np[i, 0], bounds_np[i, 0]],
                        [bounds_np[0, 1], bounds_np[1, 1]],
                        [bounds_np[j, 2], bounds_np[j, 2]],
                        'gray', alpha=0.5
                    )
                    ax.plot3D(
                        [bounds_np[i, 0], bounds_np[i, 0]],
                        [bounds_np[j, 1], bounds_np[j, 1]],
                        [bounds_np[0, 2], bounds_np[1, 2]],
                        'gray', alpha=0.5
                    )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('FW-H Setup')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if interactive:
            plt.show()

        return None
