"""Integration tests for deployment batch-analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from codec.exceptions import CodecConfigurationError
from interfaces.deployment import (
    create_deployment_service,
    evaluate_deployment_batch,
    load_deployment_artifacts,
    select_gallery_frames,
)
from pipelines.training import TrainingExperimentConfig

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


def test_deployment_loader_rejects_stale_export_config(tmp_path: Path) -> None:
    """Deployment loading should fail when copied YAML drifts from the summary config."""
    export_dir = tmp_path / "demo"
    runtime_asset_dir = export_dir / "runtime_assets"
    runtime_asset_dir.mkdir(parents=True)
    (export_dir / "encoder.onnx").write_bytes(b"onnx")

    config = TrainingExperimentConfig.from_yaml(Path("configs/experiments/demo.yaml"))
    summary_payload = {
        "experiment_config": config.to_dict(),
        "best_checkpoint_path": "models/checkpoints/demo/best.pt",
    }
    (export_dir / "training_summary.json").write_text(
        json.dumps(summary_payload),
        encoding="utf-8",
    )
    (runtime_asset_dir / "runtime_config.json").write_text(
        json.dumps(config.to_dict()["runtime"]),
        encoding="utf-8",
    )

    stale_config_payload = config.to_dict()
    stale_config_payload["training"]["batch_size"] = 1
    (export_dir / "demo.yaml").write_text(
        yaml.safe_dump(stale_config_payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(CodecConfigurationError, match="stale"):
        load_deployment_artifacts(export_dir)
