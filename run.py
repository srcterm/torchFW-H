#!/usr/bin/env python3
"""Command-line interface for PyTorch FW-H aeroacoustics solver."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def cmd_preview(args):
    """Run preview command."""
    from src.utils.config import load_config, validate_config
    from src.utils.preview import preview, test_interpolation

    # Load and validate config
    config = load_config(args.config)
    issues = validate_config(config)

    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        if not args.force:
            print("\nUse --force to continue anyway.")
            return 1

    # Run preview
    result = preview(
        config=config,
        interactive=not args.no_interactive,
        save_path=args.save
    )

    # Optionally test interpolation
    if args.test_interp:
        test_interpolation(config, loader=result['loader'])

    return 0


def cmd_info(args):
    """Show information about data files."""
    from src.loaders.xdmf import XDMFLoader

    path = Path(args.path)

    if path.suffix.lower() == '.xdmf' or path.is_dir():
        loader = XDMFLoader(path)
        meta = loader.metadata

        print(f"\nXDMF Data Info: {path}")
        print("=" * 50)
        print(f"Points per snapshot: {meta.n_points:,}")
        print(f"Timesteps: {meta.n_timesteps}")
        print(f"dt: {meta.dt:.6e} s", end="")
        if not meta.uniform_dt:
            print(" (non-uniform)")
        else:
            print()
        print(f"Time range: [{meta.times[0].item():.6f}, {meta.times[-1].item():.6f}] s")
        print(f"Duration: {meta.duration:.6f} s")
        print(f"Fields: {meta.field_names}")
        print(f"Bounds:")
        print(f"  min: {meta.bounds[0].tolist()}")
        print(f"  max: {meta.bounds[1].tolist()}")

        # Load first snapshot and show field stats
        if args.stats:
            print("\nField statistics (first snapshot):")
            snapshot = loader.get_snapshot(0)
            for name, field in snapshot.fields.items():
                print(f"  {name}: min={field.min().item():.4e}, "
                      f"max={field.max().item():.4e}, "
                      f"mean={field.mean().item():.4e}")
    else:
        print(f"Unknown file type: {path}")
        return 1

    return 0


def cmd_surface(args):
    """Generate and visualize a surface."""
    from src.surfaces.parametric import cylinder, sphere, box
    from src.postprocessing.plots import plot_surface

    if args.type == 'cylinder':
        surface = cylinder(
            radius=args.radius,
            length=args.length,
            center=(0, 0, 0),
            n_theta=args.n_theta,
            n_z=args.n_z,
            caps=not args.no_caps
        )
    elif args.type == 'sphere':
        surface = sphere(
            radius=args.radius,
            center=(0, 0, 0),
            n_theta=args.n_theta,
            n_phi=args.n_phi
        )
    elif args.type == 'box':
        surface = box(
            extents=(args.lx, args.ly, args.lz),
            center=(0, 0, 0),
            n_per_side=args.n_per_side
        )
    else:
        print(f"Unknown surface type: {args.type}")
        return 1

    print(f"\n{args.type.capitalize()} Surface:")
    print(f"  Points: {surface.n_points:,}")
    print(f"  Total area: {surface.total_area:.4f}")
    print(f"  Mean spacing: {surface.mean_spacing:.6f}")

    if not args.no_plot:
        plotter = plot_surface(surface, show_normals=args.normals)
        if plotter is not None:
            plotter.show()

    return 0


def cmd_test_interp_all(args):
    """Test interpolation of all FW-H fields across all timesteps."""
    from src.utils.config import load_config, validate_config
    from src.utils.preview import test_interpolation_all

    config = load_config(args.config)
    issues = validate_config(config)

    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        if not args.force:
            print("\nUse --force to continue anyway.")
            return 1

    test_interpolation_all(
        config=config,
        save_stats=args.save_stats
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch FW-H Aeroacoustics Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Preview command
    preview_parser = subparsers.add_parser(
        'preview',
        help='Preview FW-H setup (surface, observers, CFD domain)'
    )
    preview_parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to configuration JSON file'
    )
    preview_parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable interactive visualization'
    )
    preview_parser.add_argument(
        '--save', '-s',
        help='Save visualization to file'
    )
    preview_parser.add_argument(
        '--test-interp',
        action='store_true',
        help='Test interpolation from CFD to surface'
    )
    preview_parser.add_argument(
        '--force',
        action='store_true',
        help='Continue even if config validation fails'
    )

    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show information about data files'
    )
    info_parser.add_argument(
        'path',
        help='Path to data file or directory'
    )
    info_parser.add_argument(
        '--stats',
        action='store_true',
        help='Show field statistics'
    )

    # Surface command
    surface_parser = subparsers.add_parser(
        'surface',
        help='Generate and visualize a surface'
    )
    surface_parser.add_argument(
        'type',
        choices=['cylinder', 'sphere', 'box'],
        help='Surface type'
    )
    surface_parser.add_argument('--radius', '-r', type=float, default=1.0)
    surface_parser.add_argument('--length', '-l', type=float, default=2.0)
    surface_parser.add_argument('--lx', type=float, default=2.0)
    surface_parser.add_argument('--ly', type=float, default=2.0)
    surface_parser.add_argument('--lz', type=float, default=2.0)
    surface_parser.add_argument('--n-theta', type=int, default=64)
    surface_parser.add_argument('--n-z', type=int, default=32)
    surface_parser.add_argument('--n-phi', type=int, default=32)
    surface_parser.add_argument('--n-per-side', type=int, default=16)
    surface_parser.add_argument('--no-caps', action='store_true')
    surface_parser.add_argument('--normals', action='store_true', help='Show normal vectors')
    surface_parser.add_argument('--no-plot', action='store_true')

    # Test interpolation all command
    interp_all_parser = subparsers.add_parser(
        'test-interp-all',
        help='Test interpolation of FW-H fields across all timesteps'
    )
    interp_all_parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to configuration JSON file'
    )
    interp_all_parser.add_argument(
        '--save-stats',
        help='Save statistics to CSV file'
    )
    interp_all_parser.add_argument(
        '--force',
        action='store_true',
        help='Continue even if config validation fails'
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == 'preview':
        return cmd_preview(args)
    elif args.command == 'info':
        return cmd_info(args)
    elif args.command == 'surface':
        return cmd_surface(args)
    elif args.command == 'test-interp-all':
        return cmd_test_interp_all(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
