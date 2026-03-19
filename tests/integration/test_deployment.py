"""Integration tests for the deployment-facing ONNX encoder boundary."""

from __future__ import annotations

import numpy as np
import pytest

from interfaces.deployment import create_deployment_service, load_campaign_frame_sample

pytest.importorskip("onnxruntime")


def test_exported_demo_can_run_the_deployment_round_trip(
    trained_demo_artifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exported demo artifacts should encode and decode one campaign frame."""
    export_dir = (
        trained_demo_artifacts.project_root / trained_demo_artifacts.summary.export_dir
    ).resolve()

    # The deployment helpers must resolve relative dataset and checkpoint paths even
    # when the caller is outside the temporary repo root.
    monkeypatch.chdir(export_dir.parent)

    service, artifacts = create_deployment_service(export_dir)
    sample = load_campaign_frame_sample(artifacts, frame_index=0)
    evaluation = service.evaluate_frame(
        sample.frame,
        noise_floor=sample.noise_floor,
        frequency_grid_hz=sample.frequency_grid_hz,
    )
    decoded = service.decode_packet(evaluation.encode_result.packet_bytes)

    assert evaluation.encode_result.operational_bit_count > 0
    assert np.allclose(
        decoded.reconstructed_frame,
        evaluation.encode_result.reconstructed_frame,
    )
