"""Repository-level tests for the canonical manuscript demo configuration."""

from __future__ import annotations

from pathlib import Path

from pipelines.training import TrainingExperimentConfig


def test_demo_yaml_points_to_the_canonical_manuscript_run() -> None:
    """The checked-in demo config should be the only canonical first training run."""
    config = TrainingExperimentConfig.from_yaml(Path("configs/experiments/demo.yaml"))

    assert config.artifacts.experiment_name == "demo"
    assert config.artifacts.export_onnx
    assert config.dataset.source_format == "campaigns"
    assert config.dataset.dataset_path == Path("data/raw/campaigns")
    assert config.training.device == "auto"
    assert config.task is not None


def test_repo_keeps_one_single_demo_notebook() -> None:
    """The cleaned repository should expose one deployment/demo notebook only."""
    notebook_names = sorted(path.name for path in Path("notebooks").glob("*.ipynb"))

    assert notebook_names == ["demo_deploy.ipynb"]
