#!/usr/bin/env python3
"""Train the PyTorch PSD codec from a YAML experiment configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments for the training job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> int:
    """Load the experiment configuration, run training, and print the saved artifacts."""
    project_root = Path(__file__).resolve().parents[2]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from pipelines.training import TrainingExperimentConfig, run_training_experiment

    args = parse_args()
    experiment_config = TrainingExperimentConfig.from_yaml(args.config)
    summary = run_training_experiment(experiment_config, source_config_path=args.config)
    print(f"best_epoch_index: {summary.best_epoch_index}")
    print(f"best_validation_loss: {summary.best_validation_loss:.6f}")
    print(f"best_checkpoint_path: {summary.best_checkpoint_path}")
    print(f"latest_checkpoint_path: {summary.latest_checkpoint_path}")
    print(f"runtime_asset_dir: {summary.runtime_asset_dir}")
    print(f"onnx_path: {summary.onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
