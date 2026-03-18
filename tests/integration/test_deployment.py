"""Integration tests for the deployment-facing ONNX encoder boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from interfaces.deployment import create_deployment_service, load_campaign_frame_sample

pytest.importorskip("onnxruntime")


def test_exported_baseline_can_run_the_deployment_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exported baseline artifacts should encode and decode one real campaign frame."""
    export_dir = Path("models/exports/baseline_psdcodec").resolve()
    if not export_dir.exists():
        pytest.skip("The baseline deployment artifacts are not present in this workspace.")

    # The deployment helpers should resolve experiment-relative asset paths even when the
    # caller is not running from the repository root, which is how notebooks typically run.
    notebooks_dir = export_dir.parents[2] / "notebooks"
    if notebooks_dir.exists():
        monkeypatch.chdir(notebooks_dir)

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
