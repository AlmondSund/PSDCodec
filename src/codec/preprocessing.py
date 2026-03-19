"""Deterministic preprocessing and inverse preprocessing for PSD frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from codec.config import PreprocessingConfig
from codec.exceptions import CodecConfigurationError
from codec.quantization import UniformScalarQuantizer, quantize_side_information
from codec.types import PreprocessingArtifacts, QuantizedSideInformation
from utils import FloatArray, as_1d_float_array, partition_slices


def build_linear_upsampling_matrix(
    *,
    original_bin_count: int,  # Desired PSD length N
    reduced_bin_count: int,  # Reduced-resolution PSD length N_r
) -> FloatArray:
    """Build the dense interpolation matrix for the admissible linear upsampler.

    Purpose:
        Make the runtime and training-time inverse preprocessing paths share the exact
        same interpolation rule so they cannot silently drift apart.
    """
    if original_bin_count <= 0:
        raise CodecConfigurationError("original_bin_count must be strictly positive.")
    if reduced_bin_count <= 0:
        raise CodecConfigurationError("reduced_bin_count must be strictly positive.")
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


def upsample_frame_linear(
    reduced_frame: FloatArray,  # Reduced-resolution non-negative PSD frame
    *,
    original_bin_count: int,  # Desired output length N
) -> FloatArray:
    """Upsample a reduced PSD frame with the shared linear interpolation rule."""
    frame = as_1d_float_array(reduced_frame, name="reduced_frame")
    matrix = build_linear_upsampling_matrix(
        original_bin_count=original_bin_count,
        reduced_bin_count=frame.size,
    )
    return np.clip(matrix @ frame, 0.0, None)


@dataclass
class FramePreprocessor:
    """Deterministic preprocessing chain described in the manuscript."""

    config: PreprocessingConfig  # Hyperparameters defining D_r, Ψ_κ, and S_τ

    def __post_init__(self) -> None:
        """Construct quantizers once so encode and decode share exact reconstruction levels."""
        self._mean_quantizer = UniformScalarQuantizer(self.config.mean_quantizer)
        self._log_sigma_quantizer = UniformScalarQuantizer(self.config.log_sigma_quantizer)

    def preprocess(
        self,
        frame: FloatArray,  # Original non-negative PSD frame s_t
    ) -> PreprocessingArtifacts:
        """Apply downsampling, log mapping, and blockwise standardization to one frame."""
        original_frame = as_1d_float_array(frame, name="frame")
        reduced_frame = self._downsample_local_average(original_frame)
        mapped_frame = np.log(reduced_frame + self.config.dynamic_range_offset)
        block_means, block_sigmas = self._block_statistics(mapped_frame)
        normalized_frame = self._standardize(mapped_frame, block_means, block_sigmas)

        log_sigmas = np.log(block_sigmas)
        mean_codes, log_sigma_codes, quantized_means, quantized_log_sigmas = (
            quantize_side_information(
                block_means,
                log_sigmas,
                self._mean_quantizer,
                self._log_sigma_quantizer,
            )
        )
        return PreprocessingArtifacts(
            downsampled_frame=reduced_frame,
            mapped_frame=mapped_frame,
            normalized_frame=normalized_frame,
            block_means=block_means,
            block_sigmas=block_sigmas,
            side_information=QuantizedSideInformation(
                mean_codes=mean_codes,
                log_sigma_codes=log_sigma_codes,
                means=quantized_means,
                log_sigmas=quantized_log_sigmas,
            ),
        )

    def inverse_preprocess(
        self,
        normalized_frame: FloatArray,  # Normalized representation u_t or û_t
        side_information: QuantizedSideInformation,  # Quantized side information τ̂_t
        *,
        original_bin_count: int,  # Target length N after upsampling
    ) -> FloatArray:
        """Apply inverse standardization, inverse map, and upsampling to recover a PSD frame."""
        reduced_frame = as_1d_float_array(
            normalized_frame,
            name="normalized_frame",
            allow_negative=True,
        )
        expected_reduced_bins = self.config.resolve_reduced_bin_count(original_bin_count)
        if reduced_frame.size != expected_reduced_bins:
            raise CodecConfigurationError(
                "normalized_frame length does not match the preprocessing configuration.",
            )
        if side_information.means.size != self.config.block_count:
            raise CodecConfigurationError(
                "side_information block count does not match the configuration."
            )

        mapped_frame = self._inverse_standardize(
            reduced_frame,
            side_information.means,
            side_information.sigmas,
        )
        reconstructed_reduced = np.exp(mapped_frame) - self.config.dynamic_range_offset
        reconstructed_reduced = np.clip(reconstructed_reduced, 0.0, None)
        return self._upsample_linear(reconstructed_reduced, original_bin_count)

    def reconstruct_preprocessing_only(
        self,
        frame: FloatArray,  # Original non-negative PSD frame s_t
    ) -> FloatArray:
        """Reconstruct a frame using only preprocessing and quantized side information."""
        original_frame = as_1d_float_array(frame, name="frame")
        artifacts = self.preprocess(original_frame)
        return self.inverse_preprocess(
            artifacts.normalized_frame,
            artifacts.side_information,
            original_bin_count=original_frame.size,
        )

    def _downsample_local_average(
        self,
        frame: FloatArray,  # Original PSD frame
    ) -> FloatArray:
        """Downsample by averaging contiguous frequency bins."""
        reduced_bin_count = self.config.resolve_reduced_bin_count(frame.size)
        slices = partition_slices(frame.size, reduced_bin_count)
        return np.asarray([np.mean(frame[block]) for block in slices], dtype=np.float64)

    def _upsample_linear(
        self,
        reduced_frame: FloatArray,  # Reduced-resolution non-negative PSD frame
        original_bin_count: int,  # Desired output length N
    ) -> FloatArray:
        """Upsample a reduced PSD frame by linear interpolation on block centers."""
        return upsample_frame_linear(
            reduced_frame,
            original_bin_count=original_bin_count,
        )

    def _block_statistics(
        self,
        mapped_frame: FloatArray,  # Log-domain reduced frame x_t
    ) -> tuple[FloatArray, FloatArray]:
        """Compute blockwise means and stabilized standard deviations."""
        blocks = partition_slices(mapped_frame.size, self.config.block_count)
        means: list[float] = []
        sigmas: list[float] = []
        for block in blocks:
            block_values = mapped_frame[block]
            mean_value = float(np.mean(block_values))
            centered = block_values - mean_value
            variance = float(np.mean(centered * centered))
            sigma_value = float(np.sqrt(variance + self.config.stability_epsilon))
            means.append(mean_value)
            sigmas.append(sigma_value)
        return np.asarray(means, dtype=np.float64), np.asarray(sigmas, dtype=np.float64)

    def _standardize(
        self,
        mapped_frame: FloatArray,  # Log-domain reduced frame x_t
        means: FloatArray,  # Block means μ_b
        sigmas: FloatArray,  # Block standard deviations σ_b
    ) -> FloatArray:
        """Apply blockwise standardization S_τ."""
        normalized = np.empty_like(mapped_frame)
        for block_index, block in enumerate(
            partition_slices(mapped_frame.size, self.config.block_count)
        ):
            normalized[block] = (mapped_frame[block] - means[block_index]) / sigmas[block_index]
        return normalized

    def _inverse_standardize(
        self,
        normalized_frame: FloatArray,  # Standardized frame u_t
        means: FloatArray,  # Quantized or exact block means
        sigmas: FloatArray,  # Quantized or exact block standard deviations
    ) -> FloatArray:
        """Apply inverse blockwise standardization S^{-1}_{τ̂}."""
        restored = np.empty_like(normalized_frame)
        for block_index, block in enumerate(
            partition_slices(normalized_frame.size, self.config.block_count),
        ):
            restored[block] = sigmas[block_index] * normalized_frame[block] + means[block_index]
        return restored
