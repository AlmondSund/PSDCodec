"""NumPy validation and partition helpers shared across the repository."""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_MIN_LENGTH: Final[int] = 1


def as_1d_float_array(
    values: npt.ArrayLike,  # Candidate one-dimensional real-valued array
    *,  # Keyword-only validation flags keep call sites explicit
    name: str,
    allow_negative: bool = False,
) -> FloatArray:
    """Convert an input into a validated one-dimensional float64 array.

    Args:
        values: Input values that should represent a PSD-like one-dimensional signal.
        name: Human-readable name used in validation error messages.
        allow_negative: Whether negative entries are valid for this array.

    Returns:
        A float64 NumPy array with exactly one dimension.

    Raises:
        ValueError: If the input is empty, non-finite, or negative when negatives are forbidden.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array; received shape {array.shape}.")
    if array.size < _MIN_LENGTH:
        raise ValueError(f"{name} must contain at least one entry.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if not allow_negative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return array


def as_probability_vector(
    values: npt.ArrayLike,  # Candidate probability mass function
    *,  # Keyword-only name keeps error context stable
    name: str,
) -> FloatArray:
    """Convert an input into a validated probability vector.

    Args:
        values: Candidate probability mass function.
        name: Human-readable name used in validation error messages.

    Returns:
        A float64 vector whose entries are strictly positive and sum to one.

    Raises:
        ValueError: If the probabilities are invalid or degenerate.
    """
    probabilities = as_1d_float_array(values, name=name)
    if np.any(probabilities <= 0.0):
        raise ValueError(f"{name} must contain strictly positive probabilities.")
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass.")
    return probabilities / total


def partition_slices(
    length: int,  # Total number of items to partition
    part_count: int,  # Number of contiguous partitions to create
) -> tuple[slice, ...]:
    """Create nearly uniform contiguous slices covering a one-dimensional domain.

    The returned slices are guaranteed to be non-empty as long as `part_count <= length`.

    Args:
        length: Number of positions in the underlying array.
        part_count: Number of contiguous partitions.

    Returns:
        A tuple of slices covering `[0, length)` without overlap.

    Raises:
        ValueError: If the partition request is impossible.
    """
    if length < _MIN_LENGTH:
        raise ValueError("length must be positive.")
    if part_count < _MIN_LENGTH:
        raise ValueError("part_count must be positive.")
    if part_count > length:
        raise ValueError("part_count cannot exceed length because partitions must be non-empty.")

    edges = np.linspace(0, length, num=part_count + 1, dtype=np.int64)
    slices: list[slice] = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if stop <= start:
            raise ValueError("Partitioning produced an empty block; reduce part_count.")
        slices.append(slice(int(start), int(stop)))
    return tuple(slices)
