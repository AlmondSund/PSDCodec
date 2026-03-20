"""Unit tests for raw campaign ingestion and harmonization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from codec.config import PreprocessingConfig, ScalarQuantizerConfig
from codec.exceptions import CodecConfigurationError
from codec.preprocessing import FramePreprocessor
from data.campaigns import load_campaign_dataset_bundle
from data.datasets import PreparedPsdDataset


def _write_campaign(
    campaign_root: Path,
    *,
    campaign_name: str,
    campaign_id: int,
    rbw_khz: int,
    node_rows: dict[str, list[tuple[int, list[float]]]],
) -> None:
    """Write a small synthetic raw campaign directory for testing."""
    campaign_dir = campaign_root / campaign_name
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
                "campaign_label": campaign_name,
                "campaign_id": campaign_id,
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
                "rbw_kHz": rbw_khz,
                "antenna_amp": "false",
            }
        )

    for node_name, rows in node_rows.items():
        with (campaign_dir / f"{node_name}.csv").open("w", encoding="utf-8", newline="") as stream:
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
            for row_index, (timestamp_ms, pxx_values_db) in enumerate(rows, start=1):
                writer.writerow(
                    {
                        "id": row_index,
                        "mac": "00:00:00:00:00:00",
                        "campaign_id": campaign_id,
                        "pxx": json.dumps(pxx_values_db),
                        "start_freq_hz": 88_000_000,
                        "end_freq_hz": 108_000_000,
                        "timestamp": timestamp_ms,
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


def _make_preprocessor() -> FramePreprocessor:
    """Create a compact deterministic preprocessing pipeline for campaign tests."""
    return FramePreprocessor(
        PreprocessingConfig(
            reduced_bin_count=4,
            block_count=2,
            dynamic_range_offset=1.0e-6,
            stability_epsilon=1.0e-8,
            mean_quantizer=ScalarQuantizerConfig(-20.0, 20.0, 12),
            log_sigma_quantizer=ScalarQuantizerConfig(-20.0, 5.0, 12),
        )
    )


def test_campaign_loader_harmonizes_db_frames_and_computes_sequence_noise_floors(
    tmp_path: Path,
) -> None:
    """Raw campaign ingestion should convert units, resample grids, and keep sequence order."""
    campaign_root = tmp_path / "campaigns"
    _write_campaign(
        campaign_root,
        campaign_name="RBW10",
        campaign_id=1,
        rbw_khz=10,
        node_rows={
            "Node1": [
                (2_000, [0.0, -10.0, -20.0, -30.0]),
                (1_000, [-3.0, -13.0, -23.0, -33.0]),
            ]
        },
    )
    _write_campaign(
        campaign_root,
        campaign_name="RBW3",
        campaign_id=2,
        rbw_khz=3,
        node_rows={
            "Node2": [
                (3_000, [0.0, -3.0, -6.0, -9.0, -12.0, -15.0, -18.0, -21.0]),
            ]
        },
    )

    bundle = load_campaign_dataset_bundle(
        campaign_root,
        target_bin_count=4,
        noise_floor_window=2,
    )

    assert bundle.frames.shape == (3, 4)
    assert bundle.frequency_grid_hz.shape == (4,)
    assert bundle.noise_floors is not None
    assert bundle.noise_floors.shape == (3, 4)
    assert bundle.timestamps_ms is not None
    assert bundle.sequence_ids is not None
    assert np.all(bundle.frames > 0.0)
    assert np.array_equal(bundle.timestamps_ms, np.asarray([1_000, 2_000, 3_000], dtype=np.int64))
    assert np.array_equal(
        bundle.sequence_ids,
        np.asarray(["RBW10/Node1", "RBW10/Node1", "RBW3/Node2"]),
    )
    np.testing.assert_allclose(bundle.noise_floors[0], bundle.frames[0])


def test_campaign_loader_requires_a_target_bin_count_for_mixed_rbw_inputs(tmp_path: Path) -> None:
    """Mixed PSD lengths should fail fast unless a harmonized target length is configured."""
    campaign_root = tmp_path / "campaigns"
    _write_campaign(
        campaign_root,
        campaign_name="RBW10",
        campaign_id=1,
        rbw_khz=10,
        node_rows={"Node1": [(1_000, [0.0, -10.0, -20.0, -30.0])]},
    )
    _write_campaign(
        campaign_root,
        campaign_name="RBW3",
        campaign_id=2,
        rbw_khz=3,
        node_rows={"Node2": [(2_000, [0.0, -3.0, -6.0, -9.0, -12.0, -15.0, -18.0, -21.0])]},
    )

    with pytest.raises(CodecConfigurationError):
        load_campaign_dataset_bundle(campaign_root)


def test_prepared_dataset_can_be_built_directly_from_campaigns(tmp_path: Path) -> None:
    """The prepared dataset layer should accept raw campaign roots directly."""
    campaign_root = tmp_path / "campaigns"
    _write_campaign(
        campaign_root,
        campaign_name="RBW10",
        campaign_id=1,
        rbw_khz=10,
        node_rows={"Node1": [(1_000, [0.0, -10.0, -20.0, -30.0])]},
    )

    dataset = PreparedPsdDataset.from_campaigns(
        campaign_root,
        preprocessor=_make_preprocessor(),
        target_bin_count=4,
        noise_floor_window=1,
    )

    assert len(dataset) == 1
    assert dataset.original_frames.shape == (1, 4)
    assert dataset.normalized_frames.shape == (1, 4)
    assert dataset.noise_floors is not None
