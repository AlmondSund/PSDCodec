"""Integration tests for the formal demo rate-distortion-complexity evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from interfaces.evaluation import (
    demo_eval,
    render_rate_distortion_complexity_markdown,
)

pytest.importorskip("onnxruntime")


def test_formal_demo_evaluation_reports_quality_rate_runtime_and_complexity(
    trained_demo_artifacts: Any,
) -> None:
    """The evaluator should produce a complete rate-distortion-complexity report."""
    export_dir = (
        trained_demo_artifacts.project_root / trained_demo_artifacts.summary.export_dir
    ).resolve()

    report = demo_eval(
        export_dir,
        runtime_frame_count=1,
        warmup_frame_count=0,
    )

    assert report.dataset.evaluation_frame_count >= 1
    assert report.validation_reference.psd_distortion_mean >= 0.0
    assert report.validation_reference.rate_proxy_bits_mean > 0.0
    assert report.quality.psd_distortion_mean >= 0.0
    assert report.payload.operational_bits_mean > 0.0
    assert report.runtime.encode_latency_mean_ms > 0.0
    assert report.runtime.decode_latency_mean_ms > 0.0
    assert report.runtime.roundtrip_exact_fraction == pytest.approx(1.0)
    assert report.complexity.total_parameter_count > 0
    assert report.complexity.encoder_parameter_count > 0


def test_formal_demo_evaluation_markdown_and_json_are_renderable(
    trained_demo_artifacts: Any,
    tmp_path: Path,
) -> None:
    """The evaluator output should serialize cleanly for report generation."""
    export_dir = (
        trained_demo_artifacts.project_root / trained_demo_artifacts.summary.export_dir
    ).resolve()

    report = demo_eval(
        export_dir,
        runtime_frame_count=1,
        warmup_frame_count=0,
    )
    markdown = render_rate_distortion_complexity_markdown(report)
    payload = report.to_dict()
    output_path = tmp_path / "report.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    assert "Rate-Distortion-Complexity Evaluation" in markdown
    assert payload["validation_reference"]["rate_proxy_bits_mean"] > 0.0
    assert payload["complexity"]["total_parameter_count"] > 0
    persisted_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted_payload["dataset"]["evaluation_frame_count"] >= 1
