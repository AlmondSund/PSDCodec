"""Torch-compatible deterministic preprocessing helpers for training-time reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from codec.config import PreprocessingConfig
from codec.exceptions import CodecConfigurationError
from utils import partition_slices

_torch: Any | None
try:
    import torch as _torch
except ImportError:  # pragma: no cover - exercised only when torch is unavailable
    _torch = None

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


def _require_torch() -> Any:
    """Return the imported torch module or raise a precise error."""
    if _torch is None:
        raise ImportError("PyTorch is required to use codec.torch_preprocessing.")
    return _torch


@dataclass(frozen=True)
class DifferentiableInversePreprocessor:
    """Differentiable inverse preprocessing used during training-time loss evaluation."""

    config: PreprocessingConfig  # Deterministic preprocessing hyperparameters
    original_bin_count: int  # Target PSD length N
    _reduced_bin_count: int = field(init=False, repr=False)
    _block_slices: tuple[slice, ...] = field(init=False, repr=False)
    _upsampling_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate dimensions and precompute structural interpolation operators."""
        if self.original_bin_count <= 0:
            raise CodecConfigurationError("original_bin_count must be strictly positive.")
        reduced_bin_count = self.config.resolve_reduced_bin_count(self.original_bin_count)
        block_slices = partition_slices(reduced_bin_count, self.config.block_count)
        upsampling_matrix = _build_linear_upsampling_matrix(
            original_bin_count=self.original_bin_count,
            reduced_bin_count=reduced_bin_count,
        )
        object.__setattr__(self, "_reduced_bin_count", reduced_bin_count)
        object.__setattr__(self, "_block_slices", block_slices)
        object.__setattr__(self, "_upsampling_matrix", upsampling_matrix)

    @property
    def reduced_bin_count(self) -> int:
        """Return the reduced-resolution frame length N_r."""
        return self._reduced_bin_count

    def inverse_preprocess_batch(
        self,
        normalized_frames: Tensor,  # Decoder outputs û_t with shape [batch, N_r]
        block_means: Tensor,  # Quantized block means with shape [batch, B]
        block_log_sigmas: Tensor,  # Quantized block log standard deviations with shape [batch, B]
    ) -> Tensor:
        """Invert standardization, log mapping, and upsampling for one batch."""
        torch_module = _require_torch()
        if normalized_frames.ndim != 2:
            raise CodecConfigurationError(
                "normalized_frames must have shape [batch, reduced_bin_count]."
            )
        if normalized_frames.shape[1] != self.reduced_bin_count:
            raise CodecConfigurationError(
                "normalized_frames width does not match reduced_bin_count."
            )
        expected_block_shape = (normalized_frames.shape[0], self.config.block_count)
        if tuple(block_means.shape) != expected_block_shape:
            raise CodecConfigurationError("block_means must have shape [batch, block_count].")
        if tuple(block_log_sigmas.shape) != expected_block_shape:
            raise CodecConfigurationError("block_log_sigmas must have shape [batch, block_count].")

        mapped_frames = torch_module.empty_like(normalized_frames)
        sigmas = torch_module.exp(block_log_sigmas)
        for block_index, block in enumerate(self._block_slices):
            mapped_frames[:, block] = normalized_frames[:, block] * sigmas[
                :, block_index
            ].unsqueeze(1) + block_means[:, block_index].unsqueeze(1)

        reduced_frames = torch_module.exp(mapped_frames) - self.config.dynamic_range_offset
        reduced_frames = torch_module.clamp(reduced_frames, min=0.0)
        matrix = torch_module.as_tensor(
            self._upsampling_matrix,
            dtype=reduced_frames.dtype,
            device=reduced_frames.device,
        )
        return cast(Tensor, reduced_frames @ matrix.T)


def _build_linear_upsampling_matrix(
    *,
    original_bin_count: int,
    reduced_bin_count: int,
) -> np.ndarray:
    """Build a dense interpolation matrix matching the runtime linear upsampling rule."""
    if reduced_bin_count <= 0:
        raise CodecConfigurationError("reduced_bin_count must be strictly positive.")
    if original_bin_count <= 0:
        raise CodecConfigurationError("original_bin_count must be strictly positive.")
    if reduced_bin_count == 1:
        return np.ones((original_bin_count, 1), dtype=np.float64)

    slices = partition_slices(original_bin_count, reduced_bin_count)
    reduced_positions = np.asarray(
        [(block.start + block.stop - 1) / 2.0 for block in slices],
        dtype=np.float64,
    )
    weights = np.zeros((original_bin_count, reduced_bin_count), dtype=np.float64)
    for output_index in range(original_bin_count):
        position = float(output_index)
        if position <= reduced_positions[0]:
            weights[output_index, 0] = 1.0
            continue
        if position >= reduced_positions[-1]:
            weights[output_index, -1] = 1.0
            continue
        right_index = int(np.searchsorted(reduced_positions, position, side="right"))
        left_index = right_index - 1
        left_position = reduced_positions[left_index]
        right_position = reduced_positions[right_index]
        alpha = (position - left_position) / (right_position - left_position)
        weights[output_index, left_index] = 1.0 - alpha
        weights[output_index, right_index] = alpha
    return weights
