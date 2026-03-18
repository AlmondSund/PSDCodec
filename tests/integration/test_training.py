"""Integration tests for training, checkpointing, and export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codec.config import (
    CodecRuntimeConfig,
    FactorizedEntropyModelConfig,
    PreprocessingConfig,
    ScalarQuantizerConfig,
)
from models.torch_backend import TorchCodecConfig
from objectives.distortion import IllustrativeTaskConfig
from objectives.training import RateDistortionLossConfig
from pipelines.training import (
    ArtifactConfig,
    DatasetConfig,
    TorchCodecTrainer,
    TrainingConfig,
    TrainingExperimentConfig,
    load_training_checkpoint,
)

torch = pytest.importorskip("torch")


def _write_tiny_dataset(dataset_path: Path) -> None:
    """Write a deterministic toy PSD dataset to disk."""
    frames = np.asarray(
        [
            [1.0, 1.2, 2.0, 2.4, 2.0, 1.4, 1.0, 0.8],
            [1.1, 1.3, 2.2, 2.5, 2.1, 1.5, 1.1, 0.9],
            [0.9, 1.1, 1.9, 2.3, 1.9, 1.3, 0.9, 0.7],
            [1.2, 1.4, 2.1, 2.6, 2.2, 1.6, 1.2, 1.0],
            [0.8, 1.0, 1.8, 2.1, 1.8, 1.2, 0.8, 0.6],
            [1.0, 1.1, 1.95, 2.35, 1.95, 1.35, 0.95, 0.75],
        ],
        dtype=np.float64,
    )
    frequency_grid_hz = np.linspace(100.0, 107.0, num=8, dtype=np.float64)
    np.savez(dataset_path, frames=frames, frequency_grid_hz=frequency_grid_hz)


def _make_experiment_config(tmp_path: Path, *, export_onnx: bool) -> TrainingExperimentConfig:
    """Create a small CPU-only experiment configuration for smoke tests."""
    dataset_path = tmp_path / "toy_dataset.npz"
    _write_tiny_dataset(dataset_path)
    return TrainingExperimentConfig(
        dataset=DatasetConfig(
            dataset_path=dataset_path,
            noise_floor_window=2,
            validation_fraction=1.0 / 3.0,
            shuffle=True,
            seed=3,
        ),
        runtime=CodecRuntimeConfig(
            preprocessing=PreprocessingConfig(
                reduced_bin_count=4,
                block_count=2,
                dynamic_range_offset=1.0e-6,
                stability_epsilon=1.0e-8,
                mean_quantizer=ScalarQuantizerConfig(-10.0, 10.0, 12),
                log_sigma_quantizer=ScalarQuantizerConfig(-20.0, 5.0, 12),
            ),
            entropy_model=FactorizedEntropyModelConfig(alphabet_size=4, precision_bits=10),
        ),
        model=TorchCodecConfig(
            reduced_bin_count=4,
            latent_vector_count=2,
            embedding_dim=2,
            codebook_size=4,
            hidden_dim=16,
            commitment_weight=0.25,
        ),
        training=TrainingConfig(
            epoch_count=2,
            batch_size=2,
            learning_rate=5.0e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            device="cpu",
            loss=RateDistortionLossConfig(
                psd_weight=1.0,
                rate_weight=1.0e-3,
                vq_weight=1.0,
                task_weight=0.1,
            ),
        ),
        artifacts=ArtifactConfig(
            experiment_name="tiny_smoke",
            checkpoint_root=tmp_path / "checkpoints",
            export_root=tmp_path / "exports",
            export_onnx=export_onnx,
        ),
        task=IllustrativeTaskConfig(occupancy_margin=0.2, smoothing_window_bins=3),
    )


def test_training_smoke_saves_checkpoint_and_runtime_assets(tmp_path: Path) -> None:
    """A tiny training run should produce checkpoints and runtime codec assets."""
    experiment_config = _make_experiment_config(tmp_path, export_onnx=False)
    trainer = TorchCodecTrainer(experiment_config)
    training_dataset, validation_dataset = trainer.load_prepared_datasets()
    summary = trainer.fit(training_dataset, validation_dataset)

    assert summary.best_checkpoint_path is not None
    assert summary.best_checkpoint_path.exists()
    assert summary.latest_checkpoint_path is not None
    assert summary.latest_checkpoint_path.exists()
    assert (summary.runtime_asset_dir / "codebook.npy").exists()
    assert (summary.runtime_asset_dir / "entropy_probabilities.npy").exists()
    loaded = load_training_checkpoint(summary.best_checkpoint_path)
    assert loaded.experiment_config.model.codebook_size == 4
    assert loaded.metrics.epoch_index == summary.best_epoch_index


def test_training_export_writes_encoder_onnx(tmp_path: Path) -> None:
    """Training should export the encoder boundary to ONNX when requested."""
    onnx = pytest.importorskip("onnx")
    experiment_config = _make_experiment_config(tmp_path, export_onnx=True)
    trainer = TorchCodecTrainer(experiment_config)
    training_dataset, validation_dataset = trainer.load_prepared_datasets()
    summary = trainer.fit(training_dataset, validation_dataset)

    assert summary.onnx_path is not None
    assert summary.onnx_path.exists()
    model = onnx.load(str(summary.onnx_path))
    assert model.graph.input[0].name == "normalized_frame"
    assert model.graph.output[0].name == "pre_quantization_latents"
