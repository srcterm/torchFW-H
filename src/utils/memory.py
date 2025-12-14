"""Memory detection and adaptive chunking utilities for GPU/CPU workloads."""

from dataclasses import dataclass
from typing import Optional
import math

import torch


def get_available_memory(device: torch.device) -> float:
    """
    Get available memory on the specified device.

    Args:
        device: Target torch device

    Returns:
        Available memory in bytes
    """
    if device.type == 'cuda':
        # CUDA: query actual free memory
        free_mem, _ = torch.cuda.mem_get_info(device)
        return free_mem * 0.7  # Use 70% of free VRAM
    elif device.type == 'mps':
        # MPS: Use heuristic based on system RAM (shares with CPU)
        try:
            import psutil
            return psutil.virtual_memory().available * 0.5  # Conservative 50%
        except ImportError:
            # Fallback: assume 2GB available for MPS (conservative)
            return 2 * (1024**3)
    else:
        # CPU: use available RAM
        try:
            import psutil
            return psutil.virtual_memory().available * 0.6  # Use 60%
        except ImportError:
            # Fallback: assume 8GB available
            return 8 * (1024**3)


def estimate_cdist_memory(n_rows: int, n_cols: int, dtype: torch.dtype = torch.float32) -> int:
    """
    Estimate memory required for torch.cdist output plus overhead.

    Args:
        n_rows: Number of rows (target points)
        n_cols: Number of columns (source points)
        dtype: Data type for distance matrix

    Returns:
        Estimated memory in bytes
    """
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    # Distance matrix + ~50% overhead for intermediates (topk, sort, merge)
    return int(n_rows * n_cols * bytes_per_element * 1.5)


def compute_optimal_chunks(
    n_target: int,
    n_source: int,
    device: torch.device,
    memory_fraction: float = 0.8,
    min_chunk: int = 2_000,
    k: int = 8
) -> tuple[int, int]:
    """
    Compute optimal chunk sizes for k-NN search based on available memory.

    Strategy follows reference implementation:
    - Keep source_chunk SMALL (fixed range) for cache efficiency
    - Maximize target_chunk to reduce iteration count
    - Use conservative memory fractions

    Args:
        n_target: Total number of target points
        n_source: Total number of source points
        device: Target torch device
        memory_fraction: Fraction of available memory to use (after device fraction)
        min_chunk: Minimum chunk size
        k: Number of neighbors (affects merge buffer size)

    Returns:
        (target_chunk_size, source_chunk_size)
    """
    available_bytes = get_available_memory(device)
    usable_bytes = available_bytes * memory_fraction

    # Memory model for k-NN cdist with merge:
    # - cdist output: target_chunk × source_chunk × 4 bytes
    # - topk distances: target_chunk × k × 4 bytes
    # - topk indices: target_chunk × k × 8 bytes
    # - merge buffers: target_chunk × 2k × 12 bytes (dist + idx)
    # - Overhead for intermediates: ~2x total
    #
    # Dominant term is cdist: target_chunk × source_chunk × 4 bytes
    # Total ≈ target_chunk × source_chunk × 4 × 2 (with overhead)
    bytes_per_pair = 8  # 4 bytes × 2 overhead

    # Keep source_chunk in a fixed small range (like tri_chunk in reference)
    # This reduces memory fragmentation and improves cache efficiency
    source_chunk = max(min_chunk, min(20_000, int(0.02 * n_source)))

    # Compute max target_chunk given source_chunk
    target_chunk = int(usable_bytes / (source_chunk * bytes_per_pair))
    target_chunk = max(min_chunk, min(target_chunk, n_target))

    # Estimate actual memory usage
    memory_used = target_chunk * source_chunk * bytes_per_pair
    memory_used_gb = memory_used / (1024**3)

    return target_chunk, source_chunk


@dataclass
class DynamicChunker:
    """
    Adaptive chunk size manager that monitors memory pressure.

    Re-estimates available memory periodically and reduces chunk size
    if pressure is detected. Never increases for stability.

    Attributes:
        initial_target_chunk: Starting target chunk size
        initial_source_chunk: Starting source chunk size
        current_target_chunk: Current target chunk size
        current_source_chunk: Current source chunk size
        device: Target device
        recheck_interval: Re-estimate every N batches
    """
    initial_target_chunk: int
    initial_source_chunk: int
    current_target_chunk: int
    current_source_chunk: int
    device: torch.device
    recheck_interval: int = 10
    _batch_count: int = 0

    @classmethod
    def create(
        cls,
        n_target: int,
        n_source: int,
        device: torch.device,
        memory_fraction: float = 0.7,
        recheck_interval: int = 10
    ) -> 'DynamicChunker':
        """
        Create a DynamicChunker with optimal initial chunk sizes.

        Args:
            n_target: Total target points
            n_source: Total source points
            device: Target device
            memory_fraction: Fraction of memory to use
            recheck_interval: How often to re-check memory

        Returns:
            Configured DynamicChunker
        """
        target_chunk, source_chunk = compute_optimal_chunks(
            n_target, n_source, device, memory_fraction
        )
        return cls(
            initial_target_chunk=target_chunk,
            initial_source_chunk=source_chunk,
            current_target_chunk=target_chunk,
            current_source_chunk=source_chunk,
            device=device,
            recheck_interval=recheck_interval,
            _batch_count=0
        )

    def step(self, processed_target: int, total_target: int) -> tuple[int, int]:
        """
        Called each batch to potentially update chunk sizes.

        Args:
            processed_target: Number of target points already processed
            total_target: Total target points

        Returns:
            (current_target_chunk, current_source_chunk)
        """
        self._batch_count += 1

        # Only recheck periodically
        if self._batch_count % self.recheck_interval != 0:
            return self.current_target_chunk, self.current_source_chunk

        # Re-estimate based on current memory state
        remaining_target = total_target - processed_target
        if remaining_target <= 0:
            return self.current_target_chunk, self.current_source_chunk

        # Query current available memory
        available = get_available_memory(self.device)

        # Check if we should shrink (conservative: only if <50% of expected memory available)
        expected = (self.current_target_chunk * self.current_source_chunk * 4 * 1.5)
        if available < expected * 0.5:
            # Memory pressure detected - shrink by 30%
            new_target = max(256, int(self.current_target_chunk * 0.7))
            if new_target < self.current_target_chunk:
                self.current_target_chunk = new_target
                print(f"[Memory] Reduced target_chunk to {new_target:,} due to memory pressure")

        return self.current_target_chunk, self.current_source_chunk

    def get_chunks(self) -> tuple[int, int]:
        """Get current chunk sizes."""
        return self.current_target_chunk, self.current_source_chunk


def estimate_knn_subsample_size(n_points: int) -> int:
    """
    Compute subsample size for mean spacing estimation.

    Scales as sqrt(N) with bounds: min 1000, max 10000.

    Args:
        n_points: Total number of points

    Returns:
        Number of points to subsample
    """
    # sqrt(N) * 10, bounded between 1000 and 10000
    n_sample = int(math.sqrt(n_points) * 10)
    return min(n_points, max(1000, min(n_sample, 10000)))
