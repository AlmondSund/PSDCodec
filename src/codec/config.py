"""Typed configuration objects for codec-domain components."""

from __future__ import annotations

from dataclasses import dataclass

from codec.exceptions import CodecConfigurationError


@dataclass(frozen=True)
class ScalarQuantizerConfig:
    """Uniform scalar quantizer configuration for a bounded real-valued quantity."""

    minimum: float  # Inclusive lower bound of the representable range
    maximum: float  # Inclusive upper bound of the representable range
    bits: int  # Number of payload bits used per scalar

    def __post_init__(self) -> None:
        """Validate the quantizer contract."""
        if self.maximum <= self.minimum:
            raise CodecConfigurationError("Scalar quantizer maximum must be greater than minimum.")
        if self.bits <= 0:
            raise CodecConfigurationError("Scalar quantizer bits must be strictly positive.")
        if self.bits > 16:
            raise CodecConfigurationError(
                "Scalar quantizer bits above 16 are not supported by the reference bit-packer.",
            )

    @property
    def level_count(self) -> int:
        """Return the number of discrete reconstruction levels."""
        return 1 << self.bits

    @property
    def step(self) -> float:
        """Return the reconstruction step between adjacent quantization levels."""
        return (self.maximum - self.minimum) / float(self.level_count - 1)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Deterministic preprocessing hyperparameters from the manuscript."""

    reduced_bin_count: int | None = None  # Direct specification of N_r
    resolution_factor: float | None = None  # Optional ratio r used to derive N_r
    dynamic_range_offset: float = 1.0e-6  # Positive κ in log(v + κ)
    block_count: int = 4  # Number of contiguous standardization blocks
    stability_epsilon: float = 1.0e-6  # Positive ε_σ added inside the variance root
    mean_quantizer: ScalarQuantizerConfig = ScalarQuantizerConfig(-20.0, 40.0, 12)
    log_sigma_quantizer: ScalarQuantizerConfig = ScalarQuantizerConfig(-12.0, 8.0, 10)

    def __post_init__(self) -> None:
        """Validate preprocessing hyperparameters that are independent of frame length."""
        if self.reduced_bin_count is None and self.resolution_factor is None:
            raise CodecConfigurationError(
                "PreprocessingConfig requires either reduced_bin_count or resolution_factor.",
            )
        if self.reduced_bin_count is not None and self.reduced_bin_count <= 0:
            raise CodecConfigurationError("reduced_bin_count must be strictly positive.")
        if self.resolution_factor is not None and not (0.0 < self.resolution_factor <= 1.0):
            raise CodecConfigurationError("resolution_factor must lie in the interval (0, 1].")
        if self.dynamic_range_offset <= 0.0:
            raise CodecConfigurationError("dynamic_range_offset must be strictly positive.")
        if self.block_count <= 0:
            raise CodecConfigurationError("block_count must be strictly positive.")
        if self.stability_epsilon <= 0.0:
            raise CodecConfigurationError("stability_epsilon must be strictly positive.")

    def resolve_reduced_bin_count(
        self,
        original_bin_count: int,  # Length N of the original PSD frame
    ) -> int:
        """Resolve the reduced bin count N_r for a concrete frame length."""
        if original_bin_count <= 0:
            raise CodecConfigurationError("original_bin_count must be strictly positive.")
        reduced = self.reduced_bin_count
        if reduced is None:
            assert self.resolution_factor is not None
            reduced = int(self.resolution_factor * original_bin_count)
        reduced = max(1, min(int(reduced), original_bin_count))
        if self.block_count > reduced:
            raise CodecConfigurationError(
                "block_count cannot exceed the resolved reduced_bin_count.",
            )
        return reduced

    @property
    def side_information_bits_per_block(self) -> int:
        """Return the payload bits contributed by one standardization block."""
        return self.mean_quantizer.bits + self.log_sigma_quantizer.bits


@dataclass(frozen=True)
class FactorizedEntropyModelConfig:
    """Configuration for the factorized categorical entropy model."""

    alphabet_size: int  # Number of codewords J
    precision_bits: int = 12  # Frequency table precision for arithmetic coding
    pseudo_count: float = 1.0  # Additive smoothing applied when fitting probabilities

    def __post_init__(self) -> None:
        """Validate entropy-model hyperparameters."""
        if self.alphabet_size <= 1:
            raise CodecConfigurationError("alphabet_size must be at least two.")
        if not (1 <= self.precision_bits <= 15):
            raise CodecConfigurationError(
                "precision_bits must be in [1, 15] for the reference arithmetic coder.",
            )
        if self.pseudo_count <= 0.0:
            raise CodecConfigurationError("pseudo_count must be strictly positive.")


@dataclass(frozen=True)
class PacketFormatConfig:
    """Packet serialization settings for persistence and interchange."""

    magic: bytes = b"PSDC"
    version: int = 1

    def __post_init__(self) -> None:
        """Validate packet-header invariants."""
        if len(self.magic) != 4:
            raise CodecConfigurationError("Packet magic must contain exactly four bytes.")
        if not (0 <= self.version <= 255):
            raise CodecConfigurationError("Packet version must fit in one unsigned byte.")


@dataclass(frozen=True)
class CodecRuntimeConfig:
    """Top-level runtime configuration for the operational codec."""

    preprocessing: PreprocessingConfig  # Deterministic preprocessing chain
    entropy_model: FactorizedEntropyModelConfig  # Discrete symbol model for VQ indices
    packet_format: PacketFormatConfig = PacketFormatConfig()
