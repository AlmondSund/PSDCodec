"""Unit tests for factorized entropy modeling and arithmetic coding."""

from __future__ import annotations

import numpy as np

from codec.config import FactorizedEntropyModelConfig
from codec.entropy import FactorizedEntropyCodec, FactorizedEntropyModel


def test_arithmetic_coding_round_trip_recovers_the_index_sequence() -> None:
    """Arithmetic coding should be lossless for a fixed symbol model."""
    indices = np.asarray([0, 1, 2, 0, 2, 1, 0, 0, 2, 1], dtype=np.int64)
    model = FactorizedEntropyModel.from_config(
        FactorizedEntropyModelConfig(alphabet_size=3, precision_bits=10),
        probabilities=np.asarray([0.5, 0.3, 0.2], dtype=np.float64),
    )
    codec = FactorizedEntropyCodec(model)

    coded = codec.encode(indices)
    decoded = codec.decode(coded.payload, coded.bit_count, symbol_count=indices.size)

    assert np.array_equal(decoded, indices)
    assert coded.bit_count > 0
    assert model.rate_proxy(indices) > 0.0


def test_empirical_entropy_model_has_positive_probabilities() -> None:
    """Observed index histograms should produce a valid strictly positive PMF."""
    observations = np.asarray([0, 0, 1, 0, 2, 1, 0, 3], dtype=np.int64)
    model = FactorizedEntropyModel.from_observations(
        FactorizedEntropyModelConfig(alphabet_size=4, precision_bits=8, pseudo_count=0.5),
        observations,
    )

    assert np.all(model.probabilities > 0.0)
    assert np.isclose(np.sum(model.probabilities), 1.0)
