"""Unit tests for notebook-facing demo evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interfaces.evaluation import (
    EvaluationDatasetSummary,
    ModelComplexitySummary,
    PayloadCostSummary,
    RateDistortionComplexityReport,
    ReconstructionQualitySummary,
    RuntimeCostSummary,
    ValidationReferenceSummary,
    build_demo_eval_summary_rows,
    create_demo_eval_figure,
    load_demo_eval_report,
)


def _make_demo_eval_report() -> RateDistortionComplexityReport:
    """Create a small synthetic report for notebook-helper tests."""
    return RateDistortionComplexityReport(
        export_dir=Path("models/exports/demo"),
        checkpoint_path=Path("models/checkpoints/demo/best.pt"),
        onnx_provider="CPUExecutionProvider",
        validation_reference=ValidationReferenceSummary(
            summary_path=Path("models/exports/demo/training_summary.json"),
            best_epoch_index=12,
            psd_distortion_mean=0.013,
            preprocessing_distortion_mean=0.021,
            rate_proxy_bits_mean=4600.0,
            task_monitor_mean=3.2,
            deployment_score=0.81,
        ),
        dataset=EvaluationDatasetSummary(
            source_kind="raw_campaigns",
            dataset_path=Path("data/raw/campaigns"),
            evaluation_split="deployment_benchmark_subset",
            total_frame_count=64,
            evaluation_frame_count=64,
            runtime_frame_count=32,
            original_bin_count=4096,
            reduced_bin_count=1024,
            block_count=32,
            excluded_campaign_labels=("test_full_tdt",),
        ),
        quality=ReconstructionQualitySummary(
            psd_distortion_mean=0.010,
            psd_distortion_std=0.002,
            psd_distortion_min=0.007,
            psd_distortion_max=0.015,
            preprocessing_distortion_mean=0.016,
            codec_distortion_mean=0.012,
            task_distortion_mean=2.8,
        ),
        payload=PayloadCostSummary(
            operational_bits_mean=4620.0,
            operational_bits_std=11.0,
            operational_bits_min=4590,
            operational_bits_max=4640,
            side_information_bits_mean=704.0,
            index_bits_mean=3916.0,
            rate_proxy_bits_mean=4626.0,
            bits_per_original_bin_mean=1.128,
            bits_per_reduced_bin_mean=4.512,
            bits_per_latent_index_mean=9.023,
        ),
        runtime=RuntimeCostSummary(
            encode_latency_mean_ms=305.0,
            encode_latency_std_ms=74.0,
            encode_latency_min_ms=191.0,
            encode_latency_max_ms=540.0,
            decode_latency_mean_ms=176.0,
            decode_latency_std_ms=35.0,
            decode_latency_min_ms=80.0,
            decode_latency_max_ms=269.0,
            roundtrip_exact_fraction=1.0,
        ),
        complexity=ModelComplexitySummary(
            total_parameter_count=2_310_690,
            trainable_parameter_count=2_310_690,
            encoder_parameter_count=1_153_048,
            vector_quantizer_parameter_count=4_096,
            decoder_parameter_count=1_153_034,
            entropy_model_parameter_count=512,
        ),
    )


def test_load_demo_eval_report_round_trips_the_saved_json(tmp_path: Path) -> None:
    """The saved JSON artifact should reload into the typed report structure."""
    report = _make_demo_eval_report()
    report_path = tmp_path / "demo_eval.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    loaded_report = load_demo_eval_report(report_path)

    assert loaded_report.export_dir == report.export_dir
    assert loaded_report.validation_reference.best_epoch_index == 12
    assert loaded_report.dataset.excluded_campaign_labels == ("test_full_tdt",)
    assert loaded_report.payload.operational_bits_mean == pytest.approx(4620.0)


def test_build_demo_eval_summary_rows_returns_display_ready_rows() -> None:
    """The notebook summary rows should expose the main benchmark sections."""
    report = _make_demo_eval_report()

    rows = build_demo_eval_summary_rows(report)

    assert len(rows) >= 10
    assert rows[0]["section"] == "Validation"
    assert any(row["metric"] == "Benchmark operational payload" for row in rows)
    assert any(row["metric"] == "Total parameters" for row in rows)


def test_create_demo_eval_figure_returns_a_four_panel_summary() -> None:
    """The notebook helper should produce one four-panel matplotlib figure."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    report = _make_demo_eval_report()

    figure = create_demo_eval_figure(report)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 4
    assert figure.axes[0].get_title(loc="left") == "Distortion comparison"
    assert figure.axes[1].get_title(loc="left") == "Rate and payload composition"
