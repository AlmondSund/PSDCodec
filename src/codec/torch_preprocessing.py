"""Torch-compatible deterministic preprocessing helpers for training-time reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from codec.config import PreprocessingConfig
from codec.exceptions import CodecConfigurationError
from codec.preprocessing import build_linear_upsampling_matrix
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

_EXP_SAFETY_MARGIN_NATS: Final[float] = 2.0


def _require_torch() -> Any:
    """Return the imported torch module or raise a precise error."""
    if _torch is None:
        raise ImportError("PyTorch is required to use codec.torch_preprocessing.")
    return _torch


def _maximum_finite_exp_argument(
    *,
    tensor_dtype: Any,  # Torch floating-point dtype used for the inverse map
    dynamic_range_offset: float,  # Positive κ in exp(x) - κ
) -> float:
    """Return a conservative upper bound for the exponent input.

    Purpose:
        The decoder output is unconstrained, so large positive normalized values can
        make `exp(mapped_frame)` overflow during inverse preprocessing. This helper
        computes a dtype-aware saturation threshold that keeps the reconstructed PSD
        finite while preserving the nominal inverse map on all normal operating values.
    """
    torch_module = _require_torch()
    dtype_info = torch_module.finfo(tensor_dtype)
    return float(np.log(float(dtype_info.max) + dynamic_range_offset)) - _EXP_SAFETY_MARGIN_NATS


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
        upsampling_matrix = build_linear_upsampling_matrix(
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
        """Invert standardization, log mapping, and upsampling for one batch.

        Purpose:
            Reconstruct full-resolution PSD frames for training-time distortion
            evaluation while keeping the inverse log map numerically stable. Decoder
            outputs are unconstrained, so the exponent input is saturated to the
            largest safe finite value for the active tensor dtype before applying
            `exp(x) - κ`.
        """
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

        # Saturate only the positive tail so valid values remain unchanged while the
        # inverse log map cannot overflow to `inf` on unstable decoder outputs.
        max_exp_argument = _maximum_finite_exp_argument(
            tensor_dtype=mapped_frames.dtype,
            dynamic_range_offset=self.config.dynamic_range_offset,
        )
        mapped_frames = torch_module.clamp(mapped_frames, max=max_exp_argument)

        reduced_frames = torch_module.exp(mapped_frames) - self.config.dynamic_range_offset
        reduced_frames = torch_module.clamp(reduced_frames, min=0.0)
        matrix = torch_module.as_tensor(
            self._upsampling_matrix,
            dtype=reduced_frames.dtype,
            device=reduced_frames.device,
        )
        return cast(Tensor, reduced_frames @ matrix.T)
