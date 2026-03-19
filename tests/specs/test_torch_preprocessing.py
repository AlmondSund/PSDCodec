"""Unit tests for the differentiable inverse preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from codec.config import PreprocessingConfig, ScalarQuantizerConfig
from codec.preprocessing import FramePreprocessor
from codec.torch_preprocessing import DifferentiableInversePreprocessor


def _make_config() -> PreprocessingConfig:
    """Create a compact preprocessing config shared by the inverse-preprocessing tests."""
    return PreprocessingConfig(
        reduced_bin_count=4,
        block_count=2,
        dynamic_range_offset=1.0e-6,
        stability_epsilon=1.0e-8,
        mean_quantizer=ScalarQuantizerConfig(-5.0, 5.0, 12),
        log_sigma_quantizer=ScalarQuantizerConfig(-20.0, 5.0, 12),
    )


def test_differentiable_inverse_preprocessing_matches_runtime_inverse() -> None:
    """Training-time inverse preprocessing should match the runtime path exactly."""
    torch = pytest.importorskip("torch")
    config = _make_config()
    preprocessor = FramePreprocessor(config)
    inverse_preprocessor = DifferentiableInversePreprocessor(config, original_bin_count=8)
    frames = np.asarray(
        [
            [1.0, 4.0, 2.0, 8.0, 3.0, 6.0, 2.0, 1.0],
            [0.5, 1.5, 3.0, 6.0, 6.5, 3.5, 1.2, 0.7],
        ],
        dtype=np.float64,
    )

    artifacts = [preprocessor.preprocess(frame) for frame in frames]
    expected = np.stack(
        [
            preprocessor.inverse_preprocess(
                artifact.normalized_frame,
                artifact.side_information,
                original_bin_count=frames.shape[1],
            )
            for artifact in artifacts
        ],
        axis=0,
    )
    reconstructed = inverse_preprocessor.inverse_preprocess_batch(
        torch.as_tensor(
            np.stack([artifact.normalized_frame for artifact in artifacts], axis=0),
            dtype=torch.float32,
        ),
        torch.as_tensor(
            np.stack([artifact.side_information.means for artifact in artifacts], axis=0),
            dtype=torch.float32,
        ),
        torch.as_tensor(
            np.stack([artifact.side_information.log_sigmas for artifact in artifacts], axis=0),
            dtype=torch.float32,
        ),
    ).detach().cpu().numpy()

    np.testing.assert_allclose(reconstructed, expected, atol=1.0e-6)
