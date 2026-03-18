"""Factorized entropy model and arithmetic-coding facade."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from codec.arithmetic import ArithmeticDecoder, ArithmeticEncoder, FrequencyTable
from codec.config import FactorizedEntropyModelConfig
from codec.exceptions import CodecConfigurationError
from codec.types import EntropyCodingResult
from utils import FloatArray, IntArray, as_probability_vector


@dataclass(frozen=True)
class FactorizedEntropyModel:
    """Static categorical entropy model for the index stream i_t."""

    probabilities: FloatArray  # Symbol probabilities q_ξ(i_m)
    precision_bits: int = 12  # Frequency-table precision for arithmetic coding
    _frequency_table: FrequencyTable = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize the categorical distribution."""
        probabilities = as_probability_vector(self.probabilities, name="probabilities")
        if probabilities.size > (1 << self.precision_bits):
            raise CodecConfigurationError(
                "precision_bits are too small to assign a positive integer count to every symbol.",
            )
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(
            self, "_frequency_table", FrequencyTable(self._probabilities_to_counts())
        )

    @classmethod
    def from_config(
        cls,
        config: FactorizedEntropyModelConfig,  # Entropy-model configuration
        probabilities: FloatArray | None = None,  # Optional explicit PMF
    ) -> FactorizedEntropyModel:
        """Build a factorized model from configuration and optional explicit probabilities."""
        if probabilities is None:
            probabilities = np.full(
                config.alphabet_size, 1.0 / config.alphabet_size, dtype=np.float64
            )
        if len(probabilities) != config.alphabet_size:
            raise CodecConfigurationError("Probability vector length does not match alphabet_size.")
        return cls(
            probabilities=np.asarray(probabilities, dtype=np.float64),
            precision_bits=config.precision_bits,
        )

    @classmethod
    def from_observations(
        cls,
        config: FactorizedEntropyModelConfig,  # Entropy-model configuration
        indices: IntArray,  # Observed VQ indices used to estimate the PMF
    ) -> FactorizedEntropyModel:
        """Fit a smoothed factorized model from empirical index counts."""
        observed = np.asarray(indices, dtype=np.int64)
        if observed.ndim != 1:
            raise CodecConfigurationError("Observed indices must be one-dimensional.")
        if np.any(observed < 0) or np.any(observed >= config.alphabet_size):
            raise CodecConfigurationError("Observed indices contain symbols outside the alphabet.")
        counts = np.full(config.alphabet_size, config.pseudo_count, dtype=np.float64)
        bincount = np.bincount(observed, minlength=config.alphabet_size).astype(np.float64)
        counts += bincount
        probabilities = counts / np.sum(counts)
        return cls(probabilities=probabilities, precision_bits=config.precision_bits)

    @property
    def alphabet_size(self) -> int:
        """Return the categorical alphabet size J."""
        return int(self.probabilities.size)

    @property
    def frequency_table(self) -> FrequencyTable:
        """Return the integer frequency table used by the arithmetic coder."""
        return self._frequency_table

    def rate_proxy(
        self,
        indices: IntArray,  # Symbol sequence i_t
    ) -> float:
        """Return the differentiable rate proxy -Σ log2 q(i_m)."""
        symbols = np.asarray(indices, dtype=np.int64)
        if symbols.ndim != 1:
            raise CodecConfigurationError("Rate proxy expects a one-dimensional symbol sequence.")
        if np.any(symbols < 0) or np.any(symbols >= self.alphabet_size):
            raise CodecConfigurationError("Rate proxy symbols are outside the entropy alphabet.")
        return float(-np.sum(np.log2(self.probabilities[symbols])))

    def _probabilities_to_counts(self) -> tuple[int, ...]:
        """Convert probabilities into a positive integer frequency table."""
        target_total = 1 << self.precision_bits
        raw = self.probabilities * float(target_total)
        counts = np.floor(raw).astype(np.int64)
        counts = np.maximum(counts, 1)

        # Adjust the integerized table to preserve the requested total mass exactly.
        difference = int(target_total - int(np.sum(counts)))
        if difference > 0:
            fractional = raw - np.floor(raw)
            order = np.argsort(-fractional)
            for index in order[:difference]:
                counts[int(index)] += 1
        elif difference < 0:
            order = np.argsort(-counts)
            remaining = -difference
            for index in order:
                while remaining > 0 and counts[int(index)] > 1:
                    counts[int(index)] -= 1
                    remaining -= 1
                if remaining == 0:
                    break
            if remaining != 0:
                raise CodecConfigurationError(
                    "Failed to fit probabilities into a positive frequency table."
                )
        return tuple(int(value) for value in counts.tolist())


@dataclass
class FactorizedEntropyCodec:
    """Arithmetic-coding facade for a factorized entropy model."""

    model: FactorizedEntropyModel  # Symbol probabilities shared by encoder and decoder

    def encode(
        self,
        indices: IntArray,  # VQ index sequence i_t
    ) -> EntropyCodingResult:
        """Arithmetic-code one symbol sequence under the factorized model."""
        symbols = np.asarray(indices, dtype=np.int64)
        if symbols.ndim != 1:
            raise CodecConfigurationError(
                "Entropy coding expects a one-dimensional symbol sequence."
            )
        if np.any(symbols < 0) or np.any(symbols >= self.model.alphabet_size):
            raise CodecConfigurationError(
                "Entropy coding symbols are outside the configured alphabet."
            )

        encoder = ArithmeticEncoder()
        for symbol in symbols.tolist():
            encoder.write(self.model.frequency_table, int(symbol))
        payload, bit_count = encoder.finish()
        return EntropyCodingResult(payload=payload, bit_count=bit_count)

    def decode(
        self,
        payload: bytes,  # Byte-aligned arithmetic bitstream
        bit_count: int,  # Number of meaningful bits inside that bitstream
        *,
        symbol_count: int,  # Number of symbols expected in the decoded sequence
    ) -> IntArray:
        """Arithmetic-decode one symbol sequence under the factorized model."""
        if symbol_count < 0:
            raise CodecConfigurationError("symbol_count must be non-negative.")
        decoder = ArithmeticDecoder(payload, bit_count)
        symbols = [decoder.read(self.model.frequency_table) for _ in range(symbol_count)]
        return np.asarray(symbols, dtype=np.int64)
