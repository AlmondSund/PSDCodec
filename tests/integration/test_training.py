"""Integration tests for training, checkpointing, and export."""

from __future__ import annotations

import csv
import json
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
    EpochProgressUpdate,
    TorchCodecTrainer,
    TrainingConfig,
    TrainingExperimentConfig,
    load_training_checkpoint,
)

torch = pytest.importorskip("torch")


def _write_tiny_campaign_dataset(campaign_root: Path) -> None:
    """Write a small raw-campaign fixture compatible with the trainer."""
    campaign_dir = campaign_root / "RBW10"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    with (campaign_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "campaign_label",
                "campaign_id",
                "start_date",
                "stop_date",
                "start_time",
                "stop_time",
                "acquisition_freq_minutes",
                "central_freq_MHz",
                "span_MHz",
                "sample_rate_hz",
                "lna_gain_dB",
                "vga_gain_dB",
                "rbw_kHz",
                "antenna_amp",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "campaign_label": "RBW10",
                "campaign_id": 99,
                "start_date": "2026-03-18T00:00:00.000Z",
                "stop_date": "2026-03-18T00:00:00.000Z",
                "start_time": "00:00:00",
                "stop_time": "01:00:00",
                "acquisition_freq_minutes": 2,
                "central_freq_MHz": 98,
                "span_MHz": 20,
                "sample_rate_hz": 20_000_000,
                "lna_gain_dB": 16,
                "vga_gain_dB": 16,
                "rbw_kHz": 10,
                "antenna_amp": "false",
            }
        )

    with (campaign_dir / "Node1.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "id",
                "mac",
                "campaign_id",
                "pxx",
                "start_freq_hz",
                "end_freq_hz",
                "timestamp",
                "lat",
                "lng",
                "excursion_peak_to_peak_hz",
                "excursion_peak_deviation_hz",
                "excursion_rms_deviation_hz",
                "depth_peak_to_peak",
                "depth_peak_deviation",
                "depth_rms_deviation",
                "created_at",
            ],
        )
        writer.writeheader()
        for row_index, frame_values_db in enumerate(
            [
                [-30.0, -25.0, -20.0, -10.0, -20.0, -25.0, -30.0, -35.0],
                [-29.0, -24.0, -19.0, -9.0, -19.0, -24.0, -29.0, -34.0],
                [-31.0, -26.0, -21.0, -11.0, -21.0, -26.0, -31.0, -36.0],
                [-28.0, -23.0, -18.0, -8.0, -18.0, -23.0, -28.0, -33.0],
                [-32.0, -27.0, -22.0, -12.0, -22.0, -27.0, -32.0, -37.0],
                [-30.5, -25.5, -20.5, -10.5, -20.5, -25.5, -30.5, -35.5],
            ],
            start=1,
        ):
            writer.writerow(
                {
                    "id": row_index,
                    "mac": "00:00:00:00:00:00",
                    "campaign_id": 99,
                    "pxx": json.dumps(frame_values_db),
                    "start_freq_hz": 88_000_000,
                    "end_freq_hz": 108_000_000,
                    "timestamp": row_index * 1_000,
                    "lat": "",
                    "lng": "",
                    "excursion_peak_to_peak_hz": "",
                    "excursion_peak_deviation_hz": "",
                    "excursion_rms_deviation_hz": "",
                    "depth_peak_to_peak": "",
                    "depth_peak_deviation": "",
                    "depth_rms_deviation": "",
                    "created_at": "2026-03-18T00:00:00.000Z",
                }
            )


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
            device="auto",
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
    pytest.importorskip("onnxscript")
    experiment_config = _make_experiment_config(tmp_path, export_onnx=True)
    trainer = TorchCodecTrainer(experiment_config)
    training_dataset, validation_dataset = trainer.load_prepared_datasets()
    summary = trainer.fit(training_dataset, validation_dataset)

    assert summary.onnx_path is not None
    assert summary.onnx_path.exists()
    model = onnx.load(str(summary.onnx_path))
    assert model.graph.input[0].name == "normalized_frame"
    assert model.graph.output[0].name == "pre_quantization_latents"


def test_training_can_load_raw_campaign_directories(tmp_path: Path) -> None:
    """The trainer should ingest raw campaign directories as a first-class dataset source."""
    campaign_root = tmp_path / "campaigns"
    _write_tiny_campaign_dataset(campaign_root)
    experiment_config = _make_experiment_config(tmp_path, export_onnx=False)
    experiment_config = TrainingExperimentConfig(
        dataset=DatasetConfig(
            dataset_path=campaign_root,
            source_format="campaigns",
            noise_floor_window=2,
            validation_fraction=1.0 / 3.0,
            shuffle=True,
            seed=3,
            campaign_target_bin_count=8,
        ),
        runtime=experiment_config.runtime,
        model=experiment_config.model,
        training=experiment_config.training,
        artifacts=experiment_config.artifacts,
        task=experiment_config.task,
    )
    trainer = TorchCodecTrainer(experiment_config)
    training_dataset, validation_dataset = trainer.load_prepared_datasets()
    summary = trainer.fit(training_dataset, validation_dataset)

    assert summary.best_checkpoint_path is not None
    assert summary.best_checkpoint_path.exists()


def test_training_reports_epoch_progress_updates(tmp_path: Path) -> None:
    """The trainer should expose one progress update per completed epoch."""
    experiment_config = _make_experiment_config(tmp_path, export_onnx=False)
    trainer = TorchCodecTrainer(experiment_config)
    training_dataset, validation_dataset = trainer.load_prepared_datasets()
    updates: list[EpochProgressUpdate] = []

    summary = trainer.fit(
        training_dataset,
        validation_dataset,
        progress_reporter=updates.append,
    )

    assert len(updates) == experiment_config.training.epoch_count
    assert updates[-1].selection_metric == experiment_config.artifacts.selection_metric
    assert updates[-1].remaining_epoch_count == 0
    assert updates[-1].completed_epoch_count == experiment_config.training.epoch_count
    assert updates[-1].best_selection_score == pytest.approx(summary.best_selection_score)
    assert updates[-1].best_validation_loss == pytest.approx(summary.best_validation_loss)
