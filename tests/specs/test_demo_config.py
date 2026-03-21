"""Repository-level tests for the canonical manuscript demo configuration."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

from pipelines.training import TrainingConfig, TrainingExperimentConfig


def _load_train_demo_module() -> ModuleType:
    """Import the demo-training entrypoint module for helper-level tests."""
    module_path = Path("scripts/jobs/train_demo.py")
    spec = importlib.util.spec_from_file_location("test_train_demo_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_yaml_points_to_the_canonical_manuscript_run() -> None:
    """The checked-in demo config should be the only canonical first training run."""
    config = TrainingExperimentConfig.from_yaml(Path("configs/experiments/demo.yaml"))

    assert config.artifacts.experiment_name == "demo"
    assert config.artifacts.export_onnx
    assert config.artifacts.selection_metric == "validation_deployment_score"
    assert config.artifacts.require_selection_to_beat_preprocessing
    assert config.dataset.source_format == "campaigns"
    assert config.dataset.dataset_path == Path("data/raw/campaigns")
    assert config.runtime.preprocessing.reduced_bin_count == 1024
    assert config.runtime.preprocessing.block_count == 32
    assert config.runtime.entropy_model.alphabet_size == 512
    assert config.model.reduced_bin_count == 1024
    assert config.model.latent_vector_count == 512
    assert config.model.codebook_size == 512
    assert config.model.hidden_dim == 128
    assert config.model.residual_block_count == 4
    assert config.model.convolution_kernel_size == 7
    assert config.training.device == "auto"
    assert config.training.batch_size == 128
    assert config.training.random_seed == 0
    assert config.training.mixed_precision == "auto"
    assert config.training.enable_model_compile
    assert config.artifacts.latest_checkpoint_interval == 10
    assert config.task is not None
    assert config.task.peak_weight == 1.0e-11
    assert config.task.peak_power_weight == 1.0
    assert config.task.centroid_weight == 0.0
    assert config.task.bandwidth_weight == 0.0


def test_repo_keeps_the_two_canonical_demo_notebooks() -> None:
    """The repository should expose the deployment and evaluation demo notebooks."""
    notebook_names = sorted(path.name for path in Path("notebooks").glob("*.ipynb"))

    assert notebook_names == ["demo_deploy.ipynb", "demo_eval.ipynb"]


def test_training_config_defaults_to_auto_device_selection() -> None:
    """Training should auto-select the best usable device unless overridden explicitly."""
    assert TrainingConfig().device == "auto"


def test_demo_dataset_cache_path_depends_on_dataset_and_preprocessing_config() -> None:
    """The demo prepared-dataset cache path should be stable for one config payload."""
    module = _load_train_demo_module()
    config = TrainingExperimentConfig.from_yaml(Path("configs/experiments/demo.yaml"))

    cache_path = module._build_demo_dataset_cache_path(
        project_root=Path("."),
        experiment_config=config,
    )

    assert cache_path.parent == Path("data/processed")
    assert cache_path.name.startswith("demo_prepared_")
    assert cache_path.suffix == ".npz"


def test_demo_dataset_cache_detects_raw_data_freshness(tmp_path: Path) -> None:
    """The demo cache should rebuild only when raw campaigns are newer."""
    module = _load_train_demo_module()
    campaign_root = tmp_path / "campaigns"
    campaign_root.mkdir()
    node_file = campaign_root / "Node1.csv"
    node_file.write_text("pxx\n[]\n", encoding="utf-8")
    cache_path = tmp_path / "prepared.npz"

    assert module._prepared_dataset_cache_is_stale(
        cache_path=cache_path,
        campaign_root=campaign_root,
    )

    cache_path.write_text("cache\n", encoding="utf-8")
    os.utime(cache_path, ns=(3_000_000_000, 3_000_000_000))
    os.utime(node_file, ns=(2_000_000_000, 2_000_000_000))
    assert not module._prepared_dataset_cache_is_stale(
        cache_path=cache_path,
        campaign_root=campaign_root,
    )

    node_file.write_text("pxx\n[1]\n", encoding="utf-8")
    os.utime(node_file, ns=(4_000_000_000, 4_000_000_000))
    assert module._prepared_dataset_cache_is_stale(
        cache_path=cache_path,
        campaign_root=campaign_root,
    )
