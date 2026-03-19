"""Shared pytest fixtures for integration tests that need a trained demo export."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

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
    TrainingSummary,
)


@dataclass(frozen=True)
class TrainedDemoArtifacts:
    """Temporary repo-like demo artifacts produced for deployment integration tests."""

    project_root: Path  # Root of the temporary repo-like layout
    summary: TrainingSummary  # Training outputs saved under `project_root`


def _write_tiny_campaign_dataset(campaign_root: Path) -> None:
    """Write a deterministic raw campaign directory for deployment-oriented tests."""
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


def _build_tiny_demo_experiment_config() -> TrainingExperimentConfig:
    """Build a small relative-path experiment config for deployment smoke tests."""
    return TrainingExperimentConfig(
        dataset=DatasetConfig(
            dataset_path=Path("data/raw/campaigns"),
            source_format="campaigns",
            noise_floor_window=2,
            validation_fraction=1.0 / 3.0,
            shuffle=True,
            seed=3,
            campaign_target_bin_count=8,
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
                task_weight=0.05,
            ),
        ),
        artifacts=ArtifactConfig(
            experiment_name="demo",
            checkpoint_root=Path("models/checkpoints"),
            export_root=Path("models/exports"),
            export_onnx=True,
        ),
        task=IllustrativeTaskConfig(
            occupancy_margin=5.0e-5,
            occupancy_temperature=2.5e-5,
            smoothing_window_bins=3,
            huber_delta=1.0e5,
            peak_weight=1.0e-12,
            centroid_weight=1.0e-12,
            bandwidth_weight=1.0e-9,
        ),
    )


@pytest.fixture(scope="session")
def trained_demo_artifacts(tmp_path_factory: pytest.TempPathFactory) -> TrainedDemoArtifacts:
    """Train one tiny demo export under a temporary repo-like layout for deployment tests."""
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")

    project_root = tmp_path_factory.mktemp("demo_project")
    campaign_root = project_root / "data" / "raw" / "campaigns"
    _write_tiny_campaign_dataset(campaign_root)

    original_cwd = Path.cwd()
    os.chdir(project_root)
    try:
        trainer = TorchCodecTrainer(_build_tiny_demo_experiment_config())
        training_dataset, validation_dataset = trainer.load_prepared_datasets()
        summary = trainer.fit(training_dataset, validation_dataset)
    finally:
        os.chdir(original_cwd)

    assert summary.onnx_path is not None
    return TrainedDemoArtifacts(project_root=project_root, summary=summary)
