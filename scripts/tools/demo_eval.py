#!/usr/bin/env python3
"""Generate a formal rate-distortion-complexity evaluation for the PSDCodec demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the formal demo evaluation job."""
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=project_root / "models" / "exports" / "demo",
        help="Path to the demo export directory. Defaults to models/exports/demo.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint override.",
    )
    parser.add_argument(
        "--onnx-provider",
        type=str,
        default="CPUExecutionProvider",
        help="ONNX Runtime provider used for encoder evaluation.",
    )
    parser.add_argument(
        "--benchmark-frame-count",
        type=int,
        default=64,
        help="Number of frames materialized for distortion/payload benchmarking.",
    )
    parser.add_argument(
        "--runtime-frame-count",
        type=int,
        default=64,
        help="Number of evaluation frames used for runtime timing.",
    )
    parser.add_argument(
        "--warmup-frame-count",
        type=int,
        default=8,
        help="Number of warmup frames evaluated before timing starts.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=project_root / "reports" / "benchmarks" / "demo_eval.md",
        help="Destination Markdown report path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=project_root / "reports" / "benchmarks" / "demo_eval.json",
        help="Destination JSON report path.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the formal demo evaluation and persist Markdown and JSON reports."""
    project_root = Path(__file__).resolve().parents[2]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from interfaces.evaluation import (
        demo_eval,
        render_rate_distortion_complexity_markdown,
    )

    args = parse_args()
    report = demo_eval(
        args.export_dir,
        checkpoint_path=args.checkpoint,
        onnx_provider=args.onnx_provider,
        benchmark_frame_count=args.benchmark_frame_count,
        runtime_frame_count=args.runtime_frame_count,
        warmup_frame_count=args.warmup_frame_count,
    )
    markdown_output = args.markdown_output
    json_output = args.json_output
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)

    markdown_output.write_text(
        render_rate_distortion_complexity_markdown(report),
        encoding="utf-8",
    )
    json_output.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )

    print(f"markdown_output: {markdown_output}")
    print(f"json_output: {json_output}")
    print(
        "heldout_validation_psd_distortion_mean: "
        f"{report.validation_reference.psd_distortion_mean:.6f}"
    )
    print(
        "heldout_validation_rate_proxy_bits_mean: "
        f"{report.validation_reference.rate_proxy_bits_mean:.3f}"
    )
    print(f"benchmark_frames: {report.dataset.total_frame_count}")
    print(f"evaluation_frames: {report.dataset.evaluation_frame_count}")
    print(f"runtime_frames: {report.dataset.runtime_frame_count}")
    print(f"mean_psd_distortion: {report.quality.psd_distortion_mean:.6f}")
    print(f"mean_operational_bits: {report.payload.operational_bits_mean:.3f}")
    print(f"mean_encode_latency_ms: {report.runtime.encode_latency_mean_ms:.3f}")
    print(f"mean_decode_latency_ms: {report.runtime.decode_latency_mean_ms:.3f}")
    print(f"total_parameter_count: {report.complexity.total_parameter_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
