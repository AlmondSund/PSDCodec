"""Structured data containers exchanged across codec-domain components."""

from __future__ import annotations

from dataclasses import dataclass

from utils import FloatArray, IntArray


@dataclass(frozen=True)
class QuantizedSideInformation:
    """Quantized block statistics transmitted alongside the index stream."""

    mean_codes: IntArray  # Quantized codes for block means μ_b
    log_sigma_codes: IntArray  # Quantized codes for block log standard deviations log σ_b
    means: FloatArray  # Dequantized block means used by the inverse standardizer
    log_sigmas: FloatArray  # Dequantized block log standard deviations

    @property
    def sigmas(self) -> FloatArray:
        """Return block standard deviations reconstructed from the quantized log scales."""
        import numpy as np

        return np.exp(self.log_sigmas)


@dataclass(frozen=True)
class PreprocessingArtifacts:
    """Intermediate results from deterministic preprocessing of one PSD frame."""

    downsampled_frame: FloatArray  # Reduced-resolution non-negative PSD frame
    mapped_frame: FloatArray  # Log-domain representation x_t
    normalized_frame: FloatArray  # Standardized representation u_t consumed by the learned codec
    block_means: FloatArray  # Unquantized block means μ_b
    block_sigmas: FloatArray  # Unquantized block standard deviations σ_b
    side_information: QuantizedSideInformation  # Quantized side information τ̂_t


@dataclass(frozen=True)
class QuantizationResult:
    """Nearest-codeword assignment output for one latent sequence."""

    indices: IntArray  # Discrete codeword indices i_t
    quantized_latents: FloatArray  # Codebook vectors selected by those indices
    squared_error: float  # Sum of squared latent-space quantization errors


@dataclass(frozen=True)
class EntropyCodingResult:
    """Arithmetic-coded representation of a discrete index sequence."""

    payload: bytes  # Byte-aligned serialized arithmetic bitstream
    bit_count: int  # Number of meaningful bits contained in payload


@dataclass(frozen=True)
class CodecPacket:
    """Self-describing packet used for storage or transmission of one encoded frame."""

    original_bin_count: int  # Original PSD length N
    reduced_bin_count: int  # Preprocessed length N_r
    block_count: int  # Number of standardization blocks B
    latent_vector_count: int  # Number of latent positions M
    side_information_payload: bytes  # Bit-packed side-information payload
    side_information_bit_count: int  # Exact transmitted bits for side information
    index_payload: bytes  # Arithmetic-coded index stream
    index_bit_count: int  # Exact arithmetic-coded length in bits

    @property
    def operational_bit_count(self) -> int:
        """Return the manuscript-aligned payload length excluding container header bytes."""
        return self.side_information_bit_count + self.index_bit_count


@dataclass(frozen=True)
class CodecEncodeResult:
    """Full output of an encode operation, including reconstruction for analysis."""

    packet: CodecPacket  # Serialized codec payload container
    packet_bytes: bytes  # Header-bearing byte representation suitable for persistence
    preprocessing: PreprocessingArtifacts  # Deterministic preprocessing intermediates
    quantization: QuantizationResult  # Vector-quantization result before entropy coding
    reconstructed_frame: FloatArray  # End-to-end reconstructed PSD frame
    preprocessing_only_frame: FloatArray  # Reconstruction using preprocessing only
    operational_bit_count: int  # Actual payload bits as defined by the codec model
    rate_proxy_bit_count: float  # Differentiable rate proxy based on the entropy model


@dataclass(frozen=True)
class CodecDecodeResult:
    """Decoded frame and metadata reconstructed from a codec packet."""

    packet: CodecPacket  # Parsed packet metadata and payload segments
    indices: IntArray  # Decoded codeword indices
    reconstructed_frame: FloatArray  # Final PSD reconstruction
