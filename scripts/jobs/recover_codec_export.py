#!/usr/bin/env python3
"""Recover a deployment export bundle from a saved training checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments for export recovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the saved checkpoint used to rebuild the export bundle.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help=(
            "Optional explicit export directory override. Defaults to the export root "
            "recorded inside the checkpointed experiment configuration."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Optional YAML configuration copied into the recovered export directory for "
            "stale-artifact detection."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Recover the export directory and print the saved artifact locations."""
    project_root = Path(__file__).resolve().parents[2]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from pipelines.training import recover_training_export_from_checkpoint

    args = parse_args()
    summary = recover_training_export_from_checkpoint(
        args.checkpoint,
        export_dir=args.export_dir,
        source_config_path=args.config,
    )
    print(f"resolved_training_device: {summary.resolved_training_device}")
    print(f"selection_metric: {summary.selection_metric}")
    print(f"best_selection_score: {summary.best_selection_score:.6f}")
    print(f"best_epoch_index: {summary.best_epoch_index}")
    print(f"best_validation_loss: {summary.best_validation_loss:.6f}")
    print(f"best_checkpoint_path: {summary.best_checkpoint_path}")
    print(f"latest_checkpoint_path: {summary.latest_checkpoint_path}")
    print(f"runtime_asset_dir: {summary.runtime_asset_dir}")
    print(f"onnx_path: {summary.onnx_path}")
    print(f"export_dir: {summary.export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
