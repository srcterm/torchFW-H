"""Device detection utility for PyTorch."""

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """
    Get the best available compute device.

    Priority: MPS (Apple Silicon) > CUDA > CPU

    Args:
        preferred: Override automatic detection. One of 'auto', 'mps', 'cuda', 'cpu'.
                   'auto' or None uses automatic detection.
                   If the preferred device is unavailable, falls back to CPU.

    Returns:
        torch.device for computation
    """
    if preferred is not None and preferred.lower() != 'auto':
        preferred = preferred.lower()
        if preferred == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        elif preferred == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif preferred == 'cpu':
            return torch.device('cpu')
        else:
            print(f"Warning: Preferred device '{preferred}' not available, falling back to CPU")
            return torch.device('cpu')

    # Automatic detection
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def device_info() -> dict:
    """
    Get information about available compute devices.

    Returns:
        Dict with device availability and properties
    """
    info = {
        'cpu': True,
        'cuda': torch.cuda.is_available(),
        'mps': torch.backends.mps.is_available(),
        'default': str(get_device()),
    }

    if info['cuda']:
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)

    return info
