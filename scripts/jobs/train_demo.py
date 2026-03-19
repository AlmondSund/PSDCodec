#!/usr/bin/env python3
"""Train the canonical manuscript-backed PSDCodec demo experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path


def _print_epoch_progress(progress_update: object) -> None:
    """Print one completed-epoch progress line for the demo training job."""
    from pipelines.training import EpochProgressUpdate

    if not isinstance(progress_update, EpochProgressUpdate):
        raise TypeError("progress_update must be an EpochProgressUpdate instance.")
    metrics = progress_update.epoch_metrics
    print(
        (
            f"epoch {progress_update.completed_epoch_count}/"
            f"{progress_update.total_epoch_count} complete | "
            f"remaining_epochs={progress_update.remaining_epoch_count} | "
            f"train_loss={metrics.training_loss:.6f} | "
            f"val_loss={metrics.validation_loss:.6f} | "
            f"best_{progress_update.selection_metric}="
            f"{progress_update.best_selection_score:.6f}"
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse optional overrides for the default demo training configuration."""
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=project_root / "configs" / "experiments" / "demo.yaml",
        type=Path,
        help="Path to the demo YAML configuration. Defaults to configs/experiments/demo.yaml.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help=(
            "Optional explicit training device override. Examples: 'cuda', 'cuda:0', 'mps', "
            "or 'cpu'."
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "Allow the demo to run on CPU when no accelerator is available. By default the "
            "demo fails fast unless it resolves to CUDA or MPS."
        ),
    )
    return parser.parse_args()


def _prepare_demo_experiment_config(
    experiment_config: object,
    *,
    project_root: Path,
    source_config_path: Path,
    device_override: str | None,
    allow_cpu: bool,
) -> object:
    """Apply demo-specific device policy before training starts.

    Purpose:
        The canonical demo is intended to exercise the learned codec on an actual
        accelerator. This helper therefore keeps CPU fallback opt-in while still
        allowing explicit device overrides for debugging and reproducibility.
    """
    from codec.preprocessing import FramePreprocessor
    from data.datasets import PreparedPsdDataset
    from pipelines.training import (
        DatasetConfig,
        TrainingExperimentConfig,
        resolve_accelerator_training_device_string,
    )

    if not isinstance(experiment_config, TrainingExperimentConfig):
        raise TypeError("experiment_config must be a TrainingExperimentConfig instance.")

    adjusted_dataset_config = experiment_config.dataset
    if adjusted_dataset_config.source_format == "campaigns":
        cache_path = _build_demo_dataset_cache_path(
            project_root=project_root,
            experiment_config=experiment_config,
        )
        if _prepared_dataset_cache_is_stale(
            cache_path=cache_path,
            campaign_root=adjusted_dataset_config.dataset_path,
            source_config_path=source_config_path,
        ):
            print(f"materializing_prepared_dataset_cache: {cache_path}", flush=True)
            prepared_dataset = PreparedPsdDataset.from_campaigns(
                adjusted_dataset_config.dataset_path,
                preprocessor=FramePreprocessor(experiment_config.runtime.preprocessing),
                include_campaign_globs=adjusted_dataset_config.campaign_include_globs,
                exclude_campaign_globs=adjusted_dataset_config.campaign_exclude_globs,
                include_node_globs=adjusted_dataset_config.campaign_node_globs,
                target_bin_count=adjusted_dataset_config.campaign_target_bin_count,
                value_scale=adjusted_dataset_config.campaign_value_scale,
                max_frames=adjusted_dataset_config.campaign_max_frames,
                noise_floor_window=adjusted_dataset_config.noise_floor_window,
                noise_floor_percentile=adjusted_dataset_config.noise_floor_percentile,
            )
            prepared_dataset.save_npz(cache_path)
            print(
                (
                    f"prepared_dataset_frames: {len(prepared_dataset)} | "
                    f"prepared_dataset_cache: {cache_path}"
                ),
                flush=True,
            )
        else:
            print(f"using_prepared_dataset_cache: {cache_path}", flush=True)
        adjusted_dataset_config = DatasetConfig(
            dataset_path=cache_path,
            source_format="npz",
            frames_key="frames",
            frequency_grid_key="frequency_grid_hz",
            noise_floor_key="noise_floors",
            noise_floor_window=None,
            noise_floor_percentile=adjusted_dataset_config.noise_floor_percentile,
            validation_fraction=adjusted_dataset_config.validation_fraction,
            shuffle=adjusted_dataset_config.shuffle,
            seed=adjusted_dataset_config.seed,
        )

    adjusted_training_config = experiment_config.training
    if device_override is not None:
        adjusted_training_config = replace(adjusted_training_config, device=device_override)

    # Resolve the accelerator up front so the demo never burns CPU time silently.
    if not allow_cpu:
        resolved_device = resolve_accelerator_training_device_string(
            adjusted_training_config.device
        )
        adjusted_training_config = replace(adjusted_training_config, device=resolved_device)

    return replace(
        experiment_config,
        dataset=adjusted_dataset_config,
        training=adjusted_training_config,
    )


def _build_demo_dataset_cache_path(
    *,
    project_root: Path,
    experiment_config: object,
) -> Path:
    """Return a stable prepared-dataset cache path for one demo configuration."""
    from pipelines.training import TrainingExperimentConfig

    if not isinstance(experiment_config, TrainingExperimentConfig):
        raise TypeError("experiment_config must be a TrainingExperimentConfig instance.")

    cache_payload = {
        "dataset": _json_safe_mapping(asdict(experiment_config.dataset)),
        "preprocessing": _json_safe_mapping(asdict(experiment_config.runtime.preprocessing)),
    }
    config_hash = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return project_root / "data" / "processed" / f"demo_prepared_{config_hash}.npz"


def _prepared_dataset_cache_is_stale(
    *,
    cache_path: Path,
    campaign_root: Path,
    source_config_path: Path,
) -> bool:
    """Return whether the prepared demo cache must be rebuilt.

    Purpose:
        The demo should reuse expensive CPU preprocessing when possible, but it must
        not silently train against stale raw campaigns or an outdated YAML config.
    """
    if not cache_path.exists():
        return True
    cache_mtime_ns = cache_path.stat().st_mtime_ns
    if source_config_path.stat().st_mtime_ns > cache_mtime_ns:
        return True
    latest_campaign_mtime_ns = max(
        path.stat().st_mtime_ns
        for path in campaign_root.rglob("*")
        if path.is_file()
    )
    return latest_campaign_mtime_ns > cache_mtime_ns


def _json_safe_mapping(payload: object) -> object:
    """Convert dataclass payloads into a JSON-stable structure for cache hashing."""
    if isinstance(payload, dict):
        return {str(key): _json_safe_mapping(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_json_safe_mapping(value) for value in payload]
    if isinstance(payload, tuple):
        return [_json_safe_mapping(value) for value in payload]
    if isinstance(payload, Path):
        return str(payload)
    return payload


def main() -> int:
    """Run the demo experiment and print the saved artifact locations."""
    project_root = Path(__file__).resolve().parents[2]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from pipelines.training import TrainingExperimentConfig, run_training_experiment

    args = parse_args()
    experiment_config = TrainingExperimentConfig.from_yaml(args.config)
    experiment_config = _prepare_demo_experiment_config(
        experiment_config,
        project_root=project_root,
        source_config_path=args.config,
        device_override=args.device,
        allow_cpu=args.allow_cpu,
    )
    print(f"training_device_target: {experiment_config.training.device}", flush=True)
    summary = run_training_experiment(
        experiment_config,
        source_config_path=args.config,
        progress_reporter=_print_epoch_progress,
    )
    print(f"config_path: {args.config}")
    print(f"resolved_training_device: {summary.resolved_training_device}")
    print(f"selection_metric: {summary.selection_metric}")
    print(f"best_selection_score: {summary.best_selection_score:.6f}")
    print(f"best_epoch_index: {summary.best_epoch_index}")
    print(f"best_validation_loss: {summary.best_validation_loss:.6f}")
    print(f"best_checkpoint_path: {summary.best_checkpoint_path}")
    print(f"latest_checkpoint_path: {summary.latest_checkpoint_path}")
    print(f"runtime_asset_dir: {summary.runtime_asset_dir}")
    print(f"onnx_path: {summary.onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
