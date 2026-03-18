"""Unit tests for PSD distortion and illustrative task losses."""

from __future__ import annotations

import numpy as np

from objectives.distortion import (
    IllustrativeTaskConfig,
    illustrative_task_loss,
    log_spectral_distortion,
)


def test_log_spectral_distortion_is_zero_for_identical_frames() -> None:
    """The core PSD distortion must vanish when the frames are identical."""
    frame = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)

    distortion = log_spectral_distortion(
        frame,
        frame.copy(),
        dynamic_range_offset=1.0e-6,
    )

    assert distortion == 0.0


def test_illustrative_task_loss_is_non_negative() -> None:
    """The illustrative task regularizer should remain non-negative."""
    reference = np.asarray([0.2, 0.4, 2.0, 3.0, 0.5], dtype=np.float64)
    reconstructed = np.asarray([0.3, 0.5, 1.8, 2.5, 0.4], dtype=np.float64)
    noise_floor = np.asarray([0.1, 0.1, 0.5, 0.5, 0.1], dtype=np.float64)
    grid_hz = np.linspace(100.0, 104.0, num=5, dtype=np.float64)

    loss = illustrative_task_loss(
        reference,
        reconstructed,
        noise_floor=noise_floor,
        frequency_grid_hz=grid_hz,
        config=IllustrativeTaskConfig(occupancy_margin=0.2, smoothing_window_bins=3),
    )

    assert loss >= 0.0
