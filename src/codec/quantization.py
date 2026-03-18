"""Scalar and vector quantization primitives for PSDCodec."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from codec.config import ScalarQuantizerConfig
from codec.exceptions import CodecConfigurationError
from codec.types import QuantizationResult
from utils import FloatArray, IntArray, as_1d_float_array


@dataclass(frozen=True)
class UniformScalarQuantizer:
    """Bounded uniform scalar quantizer used for side-information payloads."""

    config: ScalarQuantizerConfig  # Quantizer bounds and bit width

    def quantize(
        self,
        values: FloatArray,  # Floating-point values to quantize
    ) -> IntArray:
        """Quantize a one-dimensional float vector into integer codes."""
        clipped = np.clip(values, self.config.minimum, self.config.maximum)
        normalized = (clipped - self.config.minimum) / self.config.step
        codes = np.rint(normalized).astype(np.int64)
        return np.clip(codes, 0, self.config.level_count - 1)

    def dequantize(
        self,
        codes: IntArray,  # Integer reconstruction levels
    ) -> FloatArray:
        """Map integer codes back to floating-point reconstruction values."""
        if np.any(codes < 0) or np.any(codes >= self.config.level_count):
            raise CodecConfigurationError("Scalar quantizer codes are outside the valid range.")
        return self.config.minimum + codes.astype(np.float64) * self.config.step


@dataclass(frozen=True)
class VectorQuantizer:
    """Nearest-neighbor vector quantizer operating on latent embeddings."""

    codebook: FloatArray  # Codebook matrix with shape [J, d]

    def __post_init__(self) -> None:
        """Validate codebook dimensionality and value range."""
        codebook = np.asarray(self.codebook, dtype=np.float64)
        if codebook.ndim != 2:
            raise CodecConfigurationError(
                f"Vector quantizer codebook must be rank 2; received shape {codebook.shape}.",
            )
        if codebook.shape[0] < 2:
            raise CodecConfigurationError(
                "Vector quantizer codebook must contain at least 2 entries."
            )
        if codebook.shape[1] < 1:
            raise CodecConfigurationError("Vector quantizer embedding dimension must be positive.")
        if not np.all(np.isfinite(codebook)):
            raise CodecConfigurationError("Vector quantizer codebook must contain finite values.")
        object.__setattr__(self, "codebook", codebook)

    @property
    def codeword_count(self) -> int:
        """Return the number of codewords J."""
        return int(self.codebook.shape[0])

    @property
    def embedding_dim(self) -> int:
        """Return the latent embedding dimension d."""
        return int(self.codebook.shape[1])

    def quantize(
        self,
        latents: FloatArray,  # Latent vectors with shape [M, d]
    ) -> QuantizationResult:
        """Assign each latent vector to its nearest codeword."""
        latent_matrix = np.asarray(latents, dtype=np.float64)
        if latent_matrix.ndim != 2:
            raise CodecConfigurationError(
                f"Latent vectors must form a rank-2 matrix; received shape {latent_matrix.shape}.",
            )
        if latent_matrix.shape[1] != self.embedding_dim:
            raise CodecConfigurationError(
                "Latent embedding dimension does not match the codebook dimension.",
            )

        # The squared Euclidean distance matrix is the exact quantity minimized by VQ.
        deltas = latent_matrix[:, np.newaxis, :] - self.codebook[np.newaxis, :, :]
        squared_distances = np.sum(deltas * deltas, axis=2)
        indices = np.argmin(squared_distances, axis=1).astype(np.int64)
        quantized = self.codebook[indices]
        min_distances = squared_distances[np.arange(indices.size), indices]
        return QuantizationResult(
            indices=indices,
            quantized_latents=quantized,
            squared_error=float(np.sum(min_distances)),
        )

    def decode(
        self,
        indices: IntArray,  # Codeword indices with shape [M]
    ) -> FloatArray:
        """Recover quantized latent vectors from their codeword indices."""
        symbol_indices = np.asarray(indices, dtype=np.int64)
        if symbol_indices.ndim != 1:
            raise CodecConfigurationError("Vector quantizer indices must be one-dimensional.")
        if np.any(symbol_indices < 0) or np.any(symbol_indices >= self.codeword_count):
            raise CodecConfigurationError(
                "Vector quantizer indices are outside the codebook range."
            )
        return self.codebook[symbol_indices]


def quantize_side_information(
    means: FloatArray,  # Unquantized block means μ_b
    log_sigmas: FloatArray,  # Unquantized block log standard deviations log σ_b
    mean_quantizer: UniformScalarQuantizer,  # Quantizer for μ_b
    log_sigma_quantizer: UniformScalarQuantizer,  # Quantizer for log σ_b
) -> tuple[IntArray, IntArray, FloatArray, FloatArray]:
    """Quantize block means and log standard deviations with explicit reconstruction values."""
    mean_values = as_1d_float_array(means, name="means", allow_negative=True)
    log_sigma_values = as_1d_float_array(log_sigmas, name="log_sigmas", allow_negative=True)
    if mean_values.size != log_sigma_values.size:
        raise CodecConfigurationError("Side-information mean and log-sigma vectors must align.")

    mean_codes = mean_quantizer.quantize(mean_values)
    log_sigma_codes = log_sigma_quantizer.quantize(log_sigma_values)
    reconstructed_means = mean_quantizer.dequantize(mean_codes)
    reconstructed_log_sigmas = log_sigma_quantizer.dequantize(log_sigma_codes)
    return mean_codes, log_sigma_codes, reconstructed_means, reconstructed_log_sigmas
