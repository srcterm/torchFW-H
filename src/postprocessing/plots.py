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

def plot_setup_slices(
        snapshot: 'CFDSnapshot',
        surface: 'PermeableSurface',
        observers: 'ObserverArray',
        fields: list[str] = None,
        slice_coords: dict = None,
        figsize: Tuple[int, int] = (15, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot 2D slice views of the FW-H setup showing levelset, surface, and observers.

        Creates 3 subplots for XY, XZ, YZ planes showing levelset field with overlaid
        FW-H surface and observer positions.

        Args:
            snapshot: CFDSnapshot containing field data (must have 'levelset')
            surface: PermeableSurface to overlay
            observers: ObserverArray to overlay
            fields: Ignored (kept for compatibility)
            slice_coords: Dict with slice coordinates {'xy': z, 'xz': y, 'yz': x}.
                            If None, uses geometry center.
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Only plot levelset
        if 'levelset' not in snapshot.fields:
            raise ValueError("snapshot must contain 'levelset' field")
        
        available_fields = ['levelset']

        points = _tensor_to_numpy(snapshot.points)
        surf_pts = _tensor_to_numpy(surface.points)
        obs_pts = _tensor_to_numpy(observers.positions)

        # Compute domain bounds
        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, z_max = points.max(axis=0)

        print(f"  CFD domain: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]")

        # Find geometry center from levelset=0
        if 'levelset' in snapshot.fields:
            levelset = _tensor_to_numpy(snapshot.fields['levelset'])
            near_zero = np.abs(levelset) < 0.1
            if near_zero.any():
                geom_pts = points[near_zero]
                geom_center = geom_pts.mean(axis=0)
                print(f"  Geometry center (from levelset): ({geom_center[0]:.2f}, {geom_center[1]:.2f}, {geom_center[2]:.2f})")
            else:
                geom_center = np.array([0,0,0]) #np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
        else:
            geom_center = np.array([0,0,0]) #np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])

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

            print(f"  {plane_name.upper()} slice: {n_points} points extracted (tol={tol:.4f})")

            u_cfd = points[mask, u_axis]
            v_cfd = points[mask, v_axis]

            # Get levelset values for this slice
            levelset_slice = _tensor_to_numpy(snapshot.fields['levelset'])[mask]
            
            print(f"    levelset: min={levelset_slice.min():.4f}, max={levelset_slice.max():.4f}")

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

            # Plot geometry (levelset < 0) as solid black
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
