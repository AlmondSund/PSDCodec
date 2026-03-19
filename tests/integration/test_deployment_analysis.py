"""Integration tests for deployment batch-analysis helpers."""

from __future__ import annotations

import pytest

from interfaces.deployment import (
    create_deployment_service,
    evaluate_deployment_batch,
    select_gallery_frames,
)

pytest.importorskip("onnxruntime")


def test_deployment_batch_analysis_reports_summary_and_gallery(
    trained_demo_artifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deployment batch-analysis helpers should summarize and sample demo frames."""
    export_dir = (
        trained_demo_artifacts.project_root / trained_demo_artifacts.summary.export_dir
    ).resolve()

    # The helpers should keep working even when notebooks are launched from elsewhere.
    monkeypatch.chdir(export_dir.parent)

    service, artifacts = create_deployment_service(export_dir)
    report = evaluate_deployment_batch(service, artifacts, max_frames=4)
    gallery = select_gallery_frames(report, gallery_size=3)

    assert report.summary.frame_count == 4
    assert report.summary.all_roundtrip_equal
    assert report.summary.packet_bits_min > 0
    assert report.summary.psd_distortion_mean >= 0.0
    assert report.assessment.verdict in {"deployment_good", "borderline", "undertrained"}
    assert len(report.frame_reports) == 4
    assert len(gallery) == 3
    assert len({frame_report.frame_index for frame_report in gallery}) == 3
