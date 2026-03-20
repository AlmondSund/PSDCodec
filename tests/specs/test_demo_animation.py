"""Unit tests for notebook-oriented deployment animation helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from interfaces.demo_animation import (
    build_animation_frame_summary_rows,
    create_deployment_animation,
    select_animation_frame_reports,
)
from interfaces.deployment import (
    DeploymentBatchReport,
    DeploymentBatchSummary,
    DeploymentFrameReport,
    DeploymentReadinessAssessment,
)


def _make_frame_report(
    *,
    frame_index: int,
    psd_distortion: float,
) -> DeploymentFrameReport:
    """Create a small synthetic deployment frame report for animation tests."""
    frequency_grid_hz = np.linspace(88.0e6, 108.0e6, num=8, dtype=np.float64)
    original_frame = np.linspace(0.1, 1.0, num=8, dtype=np.float64)
    preprocessing_frame = original_frame * (1.0 - 0.02 * (frame_index + 1))
    reconstructed_frame = original_frame * (1.0 - 0.03 * (frame_index + 1))
    noise_floor = np.full_like(original_frame, 0.05)
    return DeploymentFrameReport(
        frame_index=frame_index,
        campaign_label="RBW10",
        node_label="Node1",
        sequence_id=f"RBW10/Node1/{frame_index}",
        timestamp_ms=1_700_000_000_000 + frame_index * 1_000,
        frequency_grid_hz=frequency_grid_hz,
        original_frame=original_frame,
        preprocessing_only_frame=preprocessing_frame,
        reconstructed_frame=reconstructed_frame,
        noise_floor=noise_floor,
        operational_bit_count=112 + frame_index,
        rate_proxy_bit_count=112.5 + frame_index,
        side_information_bit_count=32,
        index_bit_count=80 + frame_index,
        psd_distortion=psd_distortion,
        preprocessing_distortion=psd_distortion * 0.8,
        codec_distortion=psd_distortion * 0.5,
        peak_frequency_error_hz=20_000.0 * (frame_index + 1),
        peak_power_error_db=0.5 + 0.1 * frame_index,
        roundtrip_equal=True,
        task_distortion=0.25 + 0.05 * frame_index,
    )


def _make_batch_report(frame_count: int = 9) -> DeploymentBatchReport:
    """Create a synthetic batch report with monotonically increasing distortion."""
    frame_reports = tuple(
        _make_frame_report(frame_index=frame_index, psd_distortion=0.01 * (frame_index + 1))
        for frame_index in range(frame_count)
    )
    summary = DeploymentBatchSummary(
        frame_count=frame_count,
        all_roundtrip_equal=True,
        packet_bits_mean=120.0,
        packet_bits_std=3.0,
        packet_bits_min=112,
        packet_bits_max=120,
        rate_proxy_bits_mean=118.0,
        rate_proxy_bits_std=2.0,
        psd_distortion_mean=0.05,
        psd_distortion_std=0.02,
        psd_distortion_min=0.01,
        psd_distortion_max=0.09,
        preprocessing_distortion_mean=0.04,
        codec_distortion_mean=0.02,
        peak_frequency_error_hz_mean=50_000.0,
        peak_frequency_error_hz_max=180_000.0,
        peak_power_error_db_mean=0.8,
        peak_power_error_db_max=1.4,
        task_distortion_mean=0.4,
    )
    assessment = DeploymentReadinessAssessment(
        verdict="deployment_good",
        reasons=("Synthetic animation test report.",),
    )
    return DeploymentBatchReport(
        frame_reports=frame_reports,
        summary=summary,
        assessment=assessment,
    )


def test_select_animation_frame_reports_spans_the_distortion_range() -> None:
    """Animation frame selection should cover the batch distortion range deterministically."""
    report = _make_batch_report(frame_count=9)

    selected_reports = select_animation_frame_reports(report, frame_count=4)

    assert len(selected_reports) == 4
    assert selected_reports[0].frame_index == 0
    assert selected_reports[-1].frame_index == 8
    assert [frame_report.frame_index for frame_report in selected_reports] == [0, 3, 5, 8]


def test_build_animation_frame_summary_rows_returns_display_ready_rows() -> None:
    """Notebook summary rows should expose the key frame-level metrics."""
    report = _make_batch_report(frame_count=3)

    rows = build_animation_frame_summary_rows(report.frame_reports)

    assert len(rows) == 3
    assert rows[0]["frame_index"] == 0
    assert rows[0]["sequence_id"] == "RBW10/Node1/0"
    assert rows[0]["packet_bits"] == 112
    assert rows[0]["peak_frequency_error_khz"] == pytest.approx(20.0)


def test_create_deployment_animation_returns_funcanimation() -> None:
    """The notebook helper should return a matplotlib FuncAnimation object."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.animation import FuncAnimation

    report = _make_batch_report(frame_count=3)

    animation = create_deployment_animation(report.frame_reports, interval_ms=250)

    assert isinstance(animation, FuncAnimation)
    assert cast(Any, animation).event_source.interval == 250


def test_create_deployment_animation_can_plot_dbm_scale() -> None:
    """The notebook animation should optionally render PSD traces in dBm."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    report = _make_batch_report(frame_count=1)
    frame_report = report.frame_reports[0]

    animation = create_deployment_animation(
        report.frame_reports,
        interval_ms=250,
        plot_dbm=True,
    )
    cast(Any, animation)._func(0)

    spectrum_axis = cast(Any, animation)._fig.axes[0]
    plotted_original = spectrum_axis.lines[0].get_ydata()

    assert spectrum_axis.get_ylabel() == "PSD [dBm]"
    np.testing.assert_allclose(
        plotted_original,
        10.0 * np.log10(frame_report.original_frame),
    )


def test_create_deployment_animation_hides_noise_floor_from_legend_when_disabled() -> None:
    """The spectrum legend should omit the noise-floor entry when disabled."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    report = _make_batch_report(frame_count=1)

    animation = create_deployment_animation(
        report.frame_reports,
        interval_ms=250,
        show_noise_floor=False,
    )

    spectrum_axis = cast(Any, animation)._fig.axes[0]
    legend = spectrum_axis.get_legend()
    legend_labels = [text.get_text() for text in legend.get_texts()]

    assert "Noise floor" not in legend_labels
