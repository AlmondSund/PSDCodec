"""Integration tests for deployment batch-analysis helpers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from codec.exceptions import CodecConfigurationError
from data.campaigns import CampaignDatasetBundle
from interfaces.deployment import (
    DeploymentArtifacts,
    create_deployment_service,
    evaluate_deployment_batch,
    load_campaign_frame_samples,
    load_deployment_artifacts,
    select_gallery_frames,
)
from pipelines.training import TrainingExperimentConfig, load_training_checkpoint

pytest.importorskip("onnxruntime")


def test_deployment_batch_analysis_reports_summary_and_gallery(
    trained_demo_artifacts: Any,
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


def test_campaign_frame_loading_falls_back_to_source_sidecar_for_npz_exports(
    trained_demo_artifacts: Any,
) -> None:
    """Deployment notebooks should recover raw campaigns from the exported sidecar.

    Purpose:
        The canonical demo export records the resolved prepared-dataset configuration
        in `training_summary.json`, which is correct for reproducible runtime loading.
        Notebook batch analysis still needs the original raw campaign root. This
        regression test keeps that fallback explicit.
    """
    export_dir = (
        trained_demo_artifacts.project_root / "models" / "exports" / "demo_npz_sidecar"
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        trained_demo_artifacts.project_root
        / "models"
        / "checkpoints"
        / "demo"
        / "best.pt"
    )
    loaded_checkpoint = load_training_checkpoint(checkpoint_path)
    source_config = loaded_checkpoint.experiment_config
    (export_dir / "demo.source.yaml").write_text(
        yaml.safe_dump(source_config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    resolved_dataset = replace(
        source_config.dataset,
        dataset_path=Path("data/processed/tiny_prepared.npz"),
        source_format="npz",
        frames_key="frames",
        frequency_grid_key="frequency_grid_hz",
        noise_floor_key="noise_floors",
        noise_floor_window=None,
    )
    artifacts = DeploymentArtifacts(
        export_dir=export_dir,
        runtime_asset_dir=export_dir / "runtime_assets",
        onnx_path=export_dir / "encoder.onnx",
        checkpoint_path=checkpoint_path,
        runtime_config=source_config.runtime,
        experiment_config=replace(source_config, dataset=resolved_dataset),
        codebook=np.zeros((1, 1), dtype=np.float64),
        probabilities=None,
    )

    samples = load_campaign_frame_samples(artifacts, max_frames=2)
    target_bin_count = loaded_checkpoint.experiment_config.dataset.campaign_target_bin_count

    assert len(samples) == 2
    assert samples[0].campaign_label == "RBW10"
    assert samples[0].frequency_grid_hz.size == target_bin_count


def test_load_campaign_frame_samples_accepts_campaign_and_node_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Notebook loaders should accept campaign/node filter overrides per call."""
    config = TrainingExperimentConfig.from_yaml(Path("configs/experiments/demo.yaml"))
    export_dir = tmp_path / "models" / "exports" / "demo"
    export_dir.mkdir(parents=True)

    observed_arguments: dict[str, object] = {}

    def _fake_load_campaign_dataset_bundle(
        campaign_root: str | Path,
        *,
        include_campaign_globs: list[str] | tuple[str, ...] | None = None,
        exclude_campaign_globs: list[str] | tuple[str, ...] | None = None,
        include_node_globs: list[str] | tuple[str, ...] | None = None,
        target_bin_count: int | None = None,
        value_scale: str = "db_to_power",
        max_frames: int | None = None,
        noise_floor_window: int | None = None,
        noise_floor_percentile: float = 10.0,
    ) -> CampaignDatasetBundle:
        del campaign_root, target_bin_count, value_scale, max_frames
        del noise_floor_window, noise_floor_percentile
        observed_arguments["include_campaign_globs"] = include_campaign_globs
        observed_arguments["exclude_campaign_globs"] = exclude_campaign_globs
        observed_arguments["include_node_globs"] = include_node_globs
        return CampaignDatasetBundle(
            frames=np.ones((1, 8), dtype=np.float64),
            frequency_grid_hz=np.linspace(88.0e6, 108.0e6, num=8, dtype=np.float64),
            campaign_labels=np.asarray(["ANTENNA_sweep"]),
            campaign_ids=np.asarray([1]),
            node_labels=np.asarray(["Node1"]),
            sequence_ids=np.asarray(["ANTENNA_sweep/Node1"]),
            timestamps_ms=np.asarray([1_700_000_000_000]),
        )

    monkeypatch.setattr(
        "interfaces.deployment.load_campaign_dataset_bundle",
        _fake_load_campaign_dataset_bundle,
    )

    artifacts = DeploymentArtifacts(
        export_dir=export_dir,
        runtime_asset_dir=export_dir / "runtime_assets",
        onnx_path=export_dir / "encoder.onnx",
        checkpoint_path=export_dir / "best.pt",
        runtime_config=config.runtime,
        experiment_config=config,
        codebook=np.zeros((1, 1), dtype=np.float64),
        probabilities=None,
    )

    samples = load_campaign_frame_samples(
        artifacts,
        max_frames=1,
        campaign_include_globs=["ANTENNA_sweep"],
        campaign_exclude_globs=["RBW_*"],
        node_include_globs=["Node1.csv"],
    )

    assert len(samples) == 1
    assert samples[0].campaign_label == "ANTENNA_sweep"
    assert samples[0].node_label == "Node1"
    assert observed_arguments["include_campaign_globs"] == ["ANTENNA_sweep"]
    assert observed_arguments["exclude_campaign_globs"] == ["RBW_*"]
    assert observed_arguments["include_node_globs"] == ["Node1.csv"]
