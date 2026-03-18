"""Unit tests for deterministic preprocessing and inverse preprocessing."""

from __future__ import annotations

import numpy as np

from codec.config import PreprocessingConfig, ScalarQuantizerConfig
from codec.preprocessing import FramePreprocessor
from codec.types import QuantizedSideInformation


def _make_config() -> PreprocessingConfig:
    """Create a compact preprocessing config for deterministic tests."""
    return PreprocessingConfig(
        reduced_bin_count=4,
        block_count=2,
        dynamic_range_offset=1.0e-6,
        stability_epsilon=1.0e-8,
        mean_quantizer=ScalarQuantizerConfig(-5.0, 5.0, 12),
        log_sigma_quantizer=ScalarQuantizerConfig(-20.0, 5.0, 12),
    )


def test_preprocessing_constant_frame_reconstructs_exactly_with_exact_side_information() -> None:
    """A constant PSD frame should remain constant under admissible preprocessing operators."""
    preprocessor = FramePreprocessor(_make_config())
    frame = np.full(8, 3.0, dtype=np.float64)

    artifacts = preprocessor.preprocess(frame)
    exact_side_information = QuantizedSideInformation(
        mean_codes=np.zeros_like(artifacts.side_information.mean_codes),
        log_sigma_codes=np.zeros_like(artifacts.side_information.log_sigma_codes),
        means=artifacts.block_means,
        log_sigmas=np.log(artifacts.block_sigmas),
    )
    reconstructed = preprocessor.inverse_preprocess(
        artifacts.normalized_frame,
        exact_side_information,
        original_bin_count=frame.size,
    )

    assert np.allclose(artifacts.normalized_frame, 0.0, atol=1.0e-12)
    assert np.allclose(reconstructed, frame, atol=1.0e-10)


def test_preprocessing_only_round_trip_stays_non_negative() -> None:
    """The preprocessing-only reconstruction should preserve shape and non-negativity."""
    preprocessor = FramePreprocessor(_make_config())
    frame = np.asarray([1.0, 4.0, 2.0, 8.0, 3.0, 6.0, 2.0, 1.0], dtype=np.float64)

    reconstructed = preprocessor.reconstruct_preprocessing_only(frame)

    assert reconstructed.shape == frame.shape
    assert np.all(reconstructed >= 0.0)
