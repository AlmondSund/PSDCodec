"""Unit tests for PSD dataset preparation and batching."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from codec.config import PreprocessingConfig, ScalarQuantizerConfig
from codec.preprocessing import FramePreprocessor
from data.datasets import PreparedPsdDataset, collate_prepared_psd_samples


def _make_preprocessor() -> FramePreprocessor:
    """Create a compact preprocessing pipeline for dataset tests."""
    return FramePreprocessor(
        PreprocessingConfig(
            reduced_bin_count=4,
            block_count=2,
            dynamic_range_offset=1.0e-6,
            stability_epsilon=1.0e-8,
            mean_quantizer=ScalarQuantizerConfig(-10.0, 10.0, 12),
            log_sigma_quantizer=ScalarQuantizerConfig(-20.0, 5.0, 12),
        )
    )


def test_prepared_dataset_precomputes_training_views_and_noise_floors() -> None:
    """Prepared datasets should expose original, normalized, and side-information arrays."""
    frames = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.5],
            [1.2, 2.1, 3.1, 4.1, 2.9, 2.2, 1.1, 0.6],
            [0.8, 1.9, 3.2, 3.8, 3.1, 2.1, 1.0, 0.4],
            [1.1, 2.3, 3.0, 4.2, 3.3, 2.4, 1.2, 0.7],
        ],
        dtype=np.float64,
    )
    dataset = PreparedPsdDataset.from_frames(
        frames,
        preprocessor=_make_preprocessor(),
        noise_floor_window=2,
    )

    assert len(dataset) == 4
    assert dataset.original_frames.shape == (4, 8)
    assert dataset.original_frames.dtype == np.float32
    assert dataset.normalized_frames.shape == (4, 4)
    assert dataset.normalized_frames.dtype == np.float32
    assert dataset.side_means.shape == (4, 2)
    assert dataset.side_means.dtype == np.float32
    assert dataset.side_log_sigmas.shape == (4, 2)
    assert dataset.side_log_sigmas.dtype == np.float32
    assert dataset.noise_floors is not None
    assert dataset.noise_floors.shape == (4, 8)
    assert dataset.noise_floors.dtype == np.float32


def test_dataset_split_and_collation_preserve_shapes() -> None:
    """Splitting and collating prepared datasets should preserve expected array shapes."""
    frames = np.tile(np.linspace(1.0, 4.0, num=8, dtype=np.float64), (6, 1))
    dataset = PreparedPsdDataset.from_frames(frames, preprocessor=_make_preprocessor())
    train_dataset, validation_dataset = dataset.train_validation_split(
        validation_fraction=1.0 / 3.0,
        seed=7,
        shuffle=True,
    )
    batch = collate_prepared_psd_samples([train_dataset[0], train_dataset[1]])

    assert len(train_dataset) == 4
    assert len(validation_dataset) == 2
    assert batch.original_frames.shape == (2, 8)
    assert batch.normalized_frames.shape == (2, 4)
    assert batch.side_means.shape == (2, 2)
    assert batch.side_log_sigmas.shape == (2, 2)


def test_prepared_dataset_npz_round_trip_reuses_precomputed_arrays(tmp_path: Path) -> None:
    """Prepared datasets saved to disk should reload without repeating preprocessing."""
    frames = np.tile(np.linspace(1.0, 4.0, num=8, dtype=np.float64), (4, 1))
    dataset = PreparedPsdDataset.from_frames(
        frames,
        preprocessor=_make_preprocessor(),
        frequency_grid_hz=np.linspace(100.0, 107.0, num=8, dtype=np.float64),
        noise_floor_window=2,
    )
    output_path = tmp_path / "prepared_dataset.npz"

    dataset.save_npz(output_path)
    loaded = PreparedPsdDataset.from_npz(output_path, preprocessor=None)

    assert loaded.original_frames.dtype == np.float32
    assert loaded.normalized_frames.dtype == np.float32
    assert loaded.side_means.dtype == np.float32
    assert loaded.side_log_sigmas.dtype == np.float32
    assert loaded.noise_floors is not None
    assert loaded.noise_floors.dtype == np.float32
    assert loaded.frequency_grid_hz is not None
    assert loaded.frequency_grid_hz.dtype == np.float64
    np.testing.assert_allclose(loaded.original_frames, dataset.original_frames)
    np.testing.assert_allclose(loaded.normalized_frames, dataset.normalized_frames)
    np.testing.assert_allclose(loaded.side_means, dataset.side_means)
    np.testing.assert_allclose(loaded.side_log_sigmas, dataset.side_log_sigmas)
    np.testing.assert_allclose(loaded.noise_floors, dataset.noise_floors)
