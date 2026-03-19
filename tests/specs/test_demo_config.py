"""Repository-level tests for the canonical manuscript demo configuration."""

from __future__ import annotations

from pathlib import Path

from pipelines.training import TrainingConfig, TrainingExperimentConfig


def test_demo_yaml_points_to_the_canonical_manuscript_run() -> None:
    """The checked-in demo config should be the only canonical first training run."""
    config = TrainingExperimentConfig.from_yaml(Path("configs/experiments/demo.yaml"))

    assert config.artifacts.experiment_name == "demo"
    assert config.artifacts.export_onnx
    assert config.artifacts.selection_metric == "validation_task_monitor"
    assert config.dataset.source_format == "campaigns"
    assert config.dataset.dataset_path == Path("data/raw/campaigns")
    assert config.runtime.preprocessing.reduced_bin_count == 512
    assert config.model.reduced_bin_count == 512
    assert config.model.latent_vector_count == 128
    assert config.training.device == "auto"
    assert config.task is not None


def test_repo_keeps_one_single_demo_notebook() -> None:
    """The cleaned repository should expose one deployment/demo notebook only."""
    notebook_names = sorted(path.name for path in Path("notebooks").glob("*.ipynb"))

    assert notebook_names == ["demo_deploy.ipynb"]


def test_training_config_defaults_to_auto_device_selection() -> None:
    """Training should auto-select the best usable device unless overridden explicitly."""
    assert TrainingConfig().device == "auto"
