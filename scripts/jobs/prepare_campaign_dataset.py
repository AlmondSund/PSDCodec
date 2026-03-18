#!/usr/bin/env python3
"""Convert raw campaign CSV acquisitions into a processed `.npz` PSD dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments for the campaign preparation job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path("data/raw/campaigns"),
        help="Root directory containing the raw campaign subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/psd_frames.npz"),
        help="Destination `.npz` file for the harmonized PSD dataset.",
    )
    parser.add_argument(
        "--target-bin-count",
        type=int,
        default=4096,
        help="Common PSD length after campaign harmonization.",
    )
    parser.add_argument(
        "--value-scale",
        choices=("db_to_power", "identity"),
        default="db_to_power",
        help="Transformation applied to the raw `pxx` values.",
    )
    parser.add_argument(
        "--noise-floor-window",
        type=int,
        default=32,
        help="Per-sequence history length used to estimate reference noise floors.",
    )
    parser.add_argument(
        "--noise-floor-percentile",
        type=float,
        default=10.0,
        help="Percentile used by the sequence-local noise-floor estimator.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional deterministic upper bound on the number of exported frames.",
    )
    parser.add_argument(
        "--include-campaign",
        action="append",
        default=None,
        help="Campaign-directory glob to include. Repeat to add more patterns.",
    )
    parser.add_argument(
        "--exclude-campaign",
        action="append",
        default=None,
        help="Campaign-directory glob to exclude. Repeat to add more patterns.",
    )
    parser.add_argument(
        "--include-node",
        action="append",
        default=None,
        help="Node-file glob to include. Repeat to add more patterns.",
    )
    return parser.parse_args()


def main() -> int:
    """Load raw campaigns, harmonize them, and persist the processed dataset."""
    project_root = Path(__file__).resolve().parents[2]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from data import load_campaign_dataset_bundle, save_campaign_dataset_bundle

    args = parse_args()
    bundle = load_campaign_dataset_bundle(
        args.campaign_root,
        include_campaign_globs=args.include_campaign,
        exclude_campaign_globs=args.exclude_campaign,
        include_node_globs=args.include_node,
        target_bin_count=args.target_bin_count,
        value_scale=args.value_scale,
        max_frames=args.max_frames,
        noise_floor_window=args.noise_floor_window,
        noise_floor_percentile=args.noise_floor_percentile,
    )
    output_path = save_campaign_dataset_bundle(bundle, args.output)
    print(f"output_path: {output_path}")
    print(f"frame_count: {bundle.frames.shape[0]}")
    print(f"bin_count: {bundle.frames.shape[1]}")
    print(f"noise_floors: {bundle.noise_floors is not None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
