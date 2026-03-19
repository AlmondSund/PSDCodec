#!/usr/bin/env python3
"""Train the canonical manuscript-backed PSDCodec demo experiment."""

from __future__ import annotations

import argparse
import sys
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
    return parser.parse_args()


def main() -> int:
    """Run the demo experiment and print the saved artifact locations."""
    project_root = Path(__file__).resolve().parents[2]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from pipelines.training import TrainingExperimentConfig, run_training_experiment

    args = parse_args()
    experiment_config = TrainingExperimentConfig.from_yaml(args.config)
    summary = run_training_experiment(
        experiment_config,
        source_config_path=args.config,
        progress_reporter=_print_epoch_progress,
    )
    print(f"config_path: {args.config}")
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
