"""Configuration loading and validation for FW-H solver."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union
import json


@dataclass
class CaseConfig:
    """Physical case parameters."""
    c0: float = 340.0  # Speed of sound (m/s)
    rho0: float = 1.225  # Reference density (kg/m³)


@dataclass
class SurfaceConfig:
    """Permeable surface configuration."""
    type: str = 'cylinder'  # 'cylinder', 'sphere', or 'box'

    # Cylinder parameters
    radius: float = 1.0
    length: float = 2.0
    axis: str = 'z'
    caps: bool = True
    n_cap_radial: Optional[int] = None

    # Sphere parameters (radius shared with cylinder)
    n_theta: int = 64
    n_phi: int = 32

    # Box parameters
    extents: Tuple[float, float, float] = (2.0, 2.0, 2.0)

    # Common parameters
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    n_z: int = 32
    n_per_side: Union[int, Tuple[int, int, int]] = 16


@dataclass
class ObserverConfig:
    """Observer array configuration."""
    type: str = 'arc'  # 'arc', 'sphere', 'line', or 'file'

    # Arc parameters
    radius: float = 100.0
    n: int = 72
    plane: str = 'xy'
    theta_range: Tuple[float, float] = (0.0, 360.0)

    # Sphere parameters
    n_theta: int = 36
    n_phi: int = 18

    # Line parameters
    start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    end: Tuple[float, float, float] = (100.0, 0.0, 0.0)

    # File parameters
    file_path: Optional[str] = None

    # Common parameters
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class DataConfig:
    """CFD data configuration."""
    format: str = 'xdmf'  # 'xdmf', 'hdf5', 'openfoam'
    path: str = ''
    fields: list = field(default_factory=lambda: ['rho', 'U', 'p'])
    field_mapping: dict = field(default_factory=dict)


@dataclass
class InterpolationConfig:
    """Interpolation settings."""
    k: int = 8  # Number of neighbors
    length_scale: Union[float, str] = 'auto'


@dataclass
class SolverConfig:
    """Solver settings."""
    f_max: Union[float, str] = 'auto'  # Cutoff frequency
    chunk_size: int = 8192  # Memory chunking size


@dataclass
class OutputConfig:
    """Output configuration."""
    path: str = './results'
    format: str = 'hdf5'  # 'hdf5' or 'csv'
    save_time_series: bool = True
    save_spectra: bool = True


@dataclass
class FWHConfig:
    """Complete FW-H solver configuration."""
    case: CaseConfig = field(default_factory=CaseConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    observers: ObserverConfig = field(default_factory=ObserverConfig)
    data: DataConfig = field(default_factory=DataConfig)
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _dict_to_dataclass(d: dict, cls):
    """Convert dict to dataclass, handling nested structures."""
    if d is None:
        return cls()

    # Filter to only include fields that exist in the dataclass
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in d.items() if k in valid_fields}

    # Convert tuples
    for key, value in filtered.items():
        if isinstance(value, list):
            field_type = cls.__dataclass_fields__[key].type
            if 'Tuple' in str(field_type):
                filtered[key] = tuple(value)

    return cls(**filtered)


def load_config(path: str | Path) -> FWHConfig:
    """
    Load configuration from JSON file.

    Args:
        path: Path to JSON configuration file

    Returns:
        FWHConfig populated from file
    """
    path = Path(path)

    with open(path, 'r') as f:
        data = json.load(f)

    config = FWHConfig(
        case=_dict_to_dataclass(data.get('case'), CaseConfig),
        surface=_dict_to_dataclass(data.get('surface'), SurfaceConfig),
        observers=_dict_to_dataclass(data.get('observers'), ObserverConfig),
        data=_dict_to_dataclass(data.get('data'), DataConfig),
        interpolation=_dict_to_dataclass(data.get('interpolation'), InterpolationConfig),
        solver=_dict_to_dataclass(data.get('solver'), SolverConfig),
        output=_dict_to_dataclass(data.get('output'), OutputConfig),
    )

    return config


def validate_config(config: FWHConfig) -> list[str]:
    """
    Validate configuration and return list of issues.

    Args:
        config: FWHConfig to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []

    # Check data path exists
    data_path = Path(config.data.path)
    if not data_path.exists():
        issues.append(f"Data path does not exist: {config.data.path}")

    # Check surface type
    if config.surface.type not in ('cylinder', 'sphere', 'box'):
        issues.append(f"Unknown surface type: {config.surface.type}")

    # Check observer type
    if config.observers.type not in ('arc', 'sphere', 'line', 'file'):
        issues.append(f"Unknown observer type: {config.observers.type}")

    # Check observer file if type is 'file'
    if config.observers.type == 'file':
        if not config.observers.file_path:
            issues.append("Observer type is 'file' but no file_path specified")
        elif not Path(config.observers.file_path).exists():
            issues.append(f"Observer file does not exist: {config.observers.file_path}")

    # Check physical parameters
    if config.case.c0 <= 0:
        issues.append(f"Speed of sound must be positive, got {config.case.c0}")
    if config.case.rho0 <= 0:
        issues.append(f"Reference density must be positive, got {config.case.rho0}")

    # Check surface dimensions
    if config.surface.radius <= 0:
        issues.append(f"Surface radius must be positive, got {config.surface.radius}")
    if config.surface.type == 'cylinder' and config.surface.length <= 0:
        issues.append(f"Cylinder length must be positive, got {config.surface.length}")

    # Check observer radius
    if config.observers.type in ('arc', 'sphere') and config.observers.radius <= 0:
        issues.append(f"Observer radius must be positive, got {config.observers.radius}")

    return issues


def save_config(config: FWHConfig, path: str | Path) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: FWHConfig to save
        path: Output file path
    """
    from dataclasses import asdict

    path = Path(path)

    data = asdict(config)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
