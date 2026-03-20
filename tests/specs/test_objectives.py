"""Unit tests for PSD distortion and illustrative task losses."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from objectives.distortion import (
    IllustrativeTaskConfig,
    build_illustrative_task_breakdown,
    extract_illustrative_features,
    hard_occupancy,
    illustrative_task_loss,
    log_spectral_distortion,
    soft_occupancy,
)
from objectives.training import torch_illustrative_task_loss


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


def test_illustrative_task_breakdown_matches_scalar_loss() -> None:
    """The public illustrative-task breakdown should agree with the scalar helper."""
    reference = np.asarray([0.2, 0.4, 2.0, 3.0, 0.5], dtype=np.float64)
    reconstructed = np.asarray([0.3, 0.5, 1.8, 2.5, 0.4], dtype=np.float64)
    noise_floor = np.asarray([0.1, 0.1, 0.5, 0.5, 0.1], dtype=np.float64)
    grid_hz = np.linspace(100.0, 104.0, num=5, dtype=np.float64)
    config = IllustrativeTaskConfig(occupancy_margin=0.2, smoothing_window_bins=3)

    breakdown = build_illustrative_task_breakdown(
        reference,
        reconstructed,
        noise_floor=noise_floor,
        frequency_grid_hz=grid_hz,
        config=config,
    )
    scalar_loss = illustrative_task_loss(
        reference,
        reconstructed,
        noise_floor=noise_floor,
        frequency_grid_hz=grid_hz,
        config=config,
    )

    assert breakdown.total_loss == scalar_loss
    assert breakdown.occupancy_loss >= 0.0
    assert breakdown.feature_loss >= 0.0
    assert breakdown.reference_soft_occupancy.shape == reference.shape
    assert breakdown.reference_hard_occupancy.dtype == np.bool_


def test_public_task_helpers_return_aligned_outputs() -> None:
    """The public occupancy and feature helpers should be shape-consistent."""
    frame = np.asarray([0.1, 0.2, 2.0, 0.3, 0.1], dtype=np.float64)
    noise_floor = np.asarray([0.05, 0.05, 0.2, 0.05, 0.05], dtype=np.float64)
    grid_hz = np.linspace(90.0e6, 94.0e6, num=5, dtype=np.float64)
    config = IllustrativeTaskConfig(occupancy_margin=0.1, smoothing_window_bins=3)

    soft_mask = soft_occupancy(frame, noise_floor=noise_floor, config=config)
    hard_mask = hard_occupancy(frame, noise_floor=noise_floor, config=config)
    features = extract_illustrative_features(
        frame,
        frequency_grid_hz=grid_hz,
        occupancy_mask=hard_mask,
        smoothing_window_bins=3,
    )

    assert soft_mask.shape == frame.shape
    assert hard_mask.shape == frame.shape
    assert 90.0e6 <= features.peak_frequency_hz <= 94.0e6
    assert np.isfinite(features.peak_power_db)


def test_soft_occupancy_remains_finite_for_large_logits() -> None:
    """Soft occupancy should saturate without emitting infinities on large margins."""
    frame = np.asarray([1.0e6, 1.0e-12], dtype=np.float64)
    noise_floor = np.asarray([0.0, 1.0e6], dtype=np.float64)
    config = IllustrativeTaskConfig(
        occupancy_margin=0.0,
        occupancy_temperature=1.0e-6,
        smoothing_window_bins=3,
    )

    soft_mask = soft_occupancy(frame, noise_floor=noise_floor, config=config)

    assert np.all(np.isfinite(soft_mask))
    assert soft_mask[0] == pytest.approx(1.0)
    assert soft_mask[1] == pytest.approx(0.0)


def test_feature_smoother_preserves_constant_spectra_at_the_edges() -> None:
    """The illustrative smoother should satisfy the manuscript constant-spectrum rule."""
    frame = np.ones(5, dtype=np.float64)
    grid_hz = np.linspace(10.0, 14.0, num=5, dtype=np.float64)
    occupancy_mask = np.ones(5, dtype=bool)

    features = extract_illustrative_features(
        frame,
        frequency_grid_hz=grid_hz,
        occupancy_mask=occupancy_mask,
        smoothing_window_bins=3,
    )

    # A constant smoothed spectrum has equal values everywhere, so `argmax` returns
    # the first index. Zero-padded smoothing would incorrectly depress the edges.
    assert features.peak_frequency_hz == pytest.approx(float(grid_hz[0]))


def test_occupied_bandwidth_uses_the_highest_mass_component() -> None:
    """Bandwidth should follow the manuscript dominant-mass component rule."""
    frame = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 10.0, 10.0], dtype=np.float64)
    grid_hz = np.linspace(100.0, 107.0, num=8, dtype=np.float64)
    occupancy_mask = np.asarray([True, True, True, True, True, False, True, True])

    features = extract_illustrative_features(
        frame,
        frequency_grid_hz=grid_hz,
        occupancy_mask=occupancy_mask,
        smoothing_window_bins=3,
    )

    # The left component is wider, but the right component has the larger integrated
    # spectral mass and therefore defines the illustrative occupied bandwidth.
    assert features.occupied_bandwidth_hz == pytest.approx(float(grid_hz[7] - grid_hz[6]))


def test_torch_illustrative_task_loss_penalizes_feature_mismatch() -> None:
    """The differentiable training surrogate should increase when features drift."""
    torch = pytest.importorskip("torch")
    config = IllustrativeTaskConfig(
        occupancy_margin=0.05,
        occupancy_temperature=0.05,
        smoothing_window_bins=3,
        huber_delta=1.0,
        peak_weight=1.0,
        centroid_weight=1.0,
        bandwidth_weight=1.0,
        occupancy_weight=1.0,
        feature_weight=1.0,
    )
    reference = torch.tensor(
        [[0.1, 0.2, 4.0, 0.2, 0.1], [0.1, 0.2, 4.0, 0.2, 0.1]],
        dtype=torch.float32,
    )
    reconstructed_good = reference.clone()
    reconstructed_bad = torch.tensor(
        [[0.1, 0.2, 4.0, 0.2, 0.1], [0.1, 4.0, 0.2, 0.2, 0.1]],
        dtype=torch.float32,
    )
    noise_floors = torch.full_like(reference, 0.05)
    frequency_grid_hz = torch.linspace(100.0, 104.0, steps=5, dtype=torch.float32)

    loss_good = torch_illustrative_task_loss(
        reference,
        reconstructed_good,
        noise_floors=noise_floors,
        frequency_grid_hz=frequency_grid_hz,
        config=config,
    )
    loss_bad = torch_illustrative_task_loss(
        reference,
        reconstructed_bad,
        noise_floors=noise_floors,
        frequency_grid_hz=frequency_grid_hz,
        config=config,
    )

    assert float(loss_bad.detach().cpu().item()) > float(loss_good.detach().cpu().item())


def test_torch_illustrative_task_loss_has_finite_gradients_for_narrow_peaks() -> None:
    """A zero-bandwidth peak should not make the training surrogate emit NaN gradients."""
    torch = pytest.importorskip("torch")
    config = IllustrativeTaskConfig(
        occupancy_margin=5.0e-6,
        occupancy_temperature=2.5e-6,
        smoothing_window_bins=5,
        huber_delta=100000.0,
        peak_weight=7.5e-13,
        centroid_weight=3.0e-12,
        bandwidth_weight=7.5e-10,
        occupancy_weight=1.0,
        feature_weight=1.0,
    )
    reference = torch.zeros((1, 4096), dtype=torch.float32)
    reference[0, 1000] = 0.2
    reconstructed = reference.clone().requires_grad_(True)
    noise_floors = torch.zeros_like(reference)
    frequency_grid_hz = torch.linspace(88.0e6, 108.0e6, steps=4096, dtype=torch.float32)

    loss = torch_illustrative_task_loss(
        reference,
        reconstructed,
        noise_floors=noise_floors,
        frequency_grid_hz=frequency_grid_hz,
        config=config,
    )
    cast(Any, loss).backward()

    assert reconstructed.grad is not None
    assert bool(torch.isfinite(reconstructed.grad).all().item())


def test_torch_illustrative_task_loss_penalizes_peak_power_mismatch() -> None:
    """The training surrogate should penalize dominant-peak power drift explicitly."""
    torch = pytest.importorskip("torch")
    config = IllustrativeTaskConfig(
        occupancy_margin=0.05,
        occupancy_temperature=0.05,
        smoothing_window_bins=3,
        huber_delta=10.0,
        peak_weight=0.0,
        peak_power_weight=1.0,
        centroid_weight=0.0,
        bandwidth_weight=0.0,
        occupancy_weight=0.0,
        feature_weight=1.0,
    )
    reference = torch.tensor([[0.1, 0.2, 4.0, 0.2, 0.1]], dtype=torch.float32)
    reconstructed_good = reference.clone()
    reconstructed_bad = torch.tensor([[0.1, 0.2, 2.0, 0.2, 0.1]], dtype=torch.float32)
    noise_floors = torch.full_like(reference, 0.05)
    frequency_grid_hz = torch.linspace(100.0, 104.0, steps=5, dtype=torch.float32)

    loss_good = torch_illustrative_task_loss(
        reference,
        reconstructed_good,
        noise_floors=noise_floors,
        frequency_grid_hz=frequency_grid_hz,
        config=config,
    )
    loss_bad = torch_illustrative_task_loss(
        reference,
        reconstructed_bad,
        noise_floors=noise_floors,
        frequency_grid_hz=frequency_grid_hz,
        config=config,
    )

    assert float(loss_bad.detach().cpu().item()) > float(loss_good.detach().cpu().item())
