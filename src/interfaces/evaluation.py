"""Formal rate-distortion-complexity evaluation helpers for the PSDCodec demo.

This module evaluates an exported demo bundle along two complementary boundaries:

- validation reference: the held-out distortion/rate metrics saved in the export,
- deployment benchmark: actual payload and runtime on a deterministic raw-frame subset,
- complexity: full learned-model parameter counts from the checkpoint.

The metric computations are kept separate from filesystem writes so scripts and
tests can reuse the same evaluation path deterministically.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal

import numpy as np

from codec.exceptions import CodecConfigurationError
from codec.preprocessing import FramePreprocessor
from data.datasets import PreparedPsdDataset
from interfaces.api import PsdCodecService
from interfaces.deployment import DeploymentArtifacts, create_deployment_service
from models.torch_backend import TorchFullCodec
from pipelines.training import TrainingExperimentConfig, load_training_checkpoint

csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class EvaluationDatasetSummary:
    """Dataset provenance and benchmark-boundary information for one evaluation."""

    source_kind: Literal["prepared_npz", "raw_campaigns"]
    dataset_path: Path  # Resolved path used to materialize the benchmark dataset
    evaluation_split: str  # Benchmark boundary label, e.g. `deployment_benchmark_subset`
    total_frame_count: int  # Frames materialized for the benchmark boundary
    evaluation_frame_count: int  # Frames actually evaluated for distortion/rate metrics
    runtime_frame_count: int  # Frames used for runtime timing
    original_bin_count: int  # Original PSD length N
    reduced_bin_count: int  # Preprocessed PSD length N_r
    block_count: int  # Number of side-information blocks B
    excluded_campaign_labels: tuple[str, ...] = ()  # Raw campaigns dropped to keep one support

    def to_dict(self) -> dict[str, str | int | tuple[str, ...]]:
        """Serialize the dataset summary into a JSON-safe mapping."""
        return {
            "source_kind": self.source_kind,
            "dataset_path": str(self.dataset_path),
            "evaluation_split": self.evaluation_split,
            "total_frame_count": self.total_frame_count,
            "evaluation_frame_count": self.evaluation_frame_count,
            "runtime_frame_count": self.runtime_frame_count,
            "original_bin_count": self.original_bin_count,
            "reduced_bin_count": self.reduced_bin_count,
            "block_count": self.block_count,
            "excluded_campaign_labels": self.excluded_campaign_labels,
        }


@dataclass(frozen=True)
class ValidationReferenceSummary:
    """Held-out validation metrics recovered from the exported training summary."""

    summary_path: Path  # `training_summary.json` source used for reference metrics
    best_epoch_index: int  # Selected epoch index recorded by the export
    psd_distortion_mean: float  # Mean validation D_psd from training-time selection
    preprocessing_distortion_mean: float  # Mean preprocessing-only validation D_psd
    rate_proxy_bits_mean: float  # Mean validation rate proxy [bits/frame]
    task_monitor_mean: float | None = None  # Optional exact task monitor on validation
    deployment_score: float | None = None  # Exported deployment-aligned selection score

    def to_dict(self) -> dict[str, str | int | float | None]:
        """Serialize the validation reference into a JSON-safe mapping."""
        return {
            "summary_path": str(self.summary_path),
            "best_epoch_index": self.best_epoch_index,
            "psd_distortion_mean": self.psd_distortion_mean,
            "preprocessing_distortion_mean": self.preprocessing_distortion_mean,
            "rate_proxy_bits_mean": self.rate_proxy_bits_mean,
            "task_monitor_mean": self.task_monitor_mean,
            "deployment_score": self.deployment_score,
        }


@dataclass(frozen=True)
class ReconstructionQualitySummary:
    """Aggregate reconstruction-quality metrics for one demo evaluation run."""

    psd_distortion_mean: float  # Mean D_psd(original, reconstructed)
    psd_distortion_std: float  # Standard deviation of D_psd
    psd_distortion_min: float  # Best observed D_psd
    psd_distortion_max: float  # Worst observed D_psd
    preprocessing_distortion_mean: float  # Mean preprocessing-only distortion
    codec_distortion_mean: float  # Mean learned-codec incremental distortion
    task_distortion_mean: float | None = None  # Optional mean illustrative-task loss

    def to_dict(self) -> dict[str, float | None]:
        """Serialize the quality summary into a JSON-safe mapping."""
        return {
            "psd_distortion_mean": self.psd_distortion_mean,
            "psd_distortion_std": self.psd_distortion_std,
            "psd_distortion_min": self.psd_distortion_min,
            "psd_distortion_max": self.psd_distortion_max,
            "preprocessing_distortion_mean": self.preprocessing_distortion_mean,
            "codec_distortion_mean": self.codec_distortion_mean,
            "task_distortion_mean": self.task_distortion_mean,
        }


@dataclass(frozen=True)
class PayloadCostSummary:
    """Aggregate bitrate- and payload-oriented metrics for one demo evaluation run."""

    operational_bits_mean: float  # Mean transmitted payload bits per frame
    operational_bits_std: float  # Standard deviation of payload bits
    operational_bits_min: int  # Minimum payload bits
    operational_bits_max: int  # Maximum payload bits
    side_information_bits_mean: float  # Mean side-information contribution
    index_bits_mean: float  # Mean entropy-coded index payload
    rate_proxy_bits_mean: float  # Mean training-time rate proxy
    bits_per_original_bin_mean: float  # Mean payload normalized by N
    bits_per_reduced_bin_mean: float  # Mean payload normalized by N_r
    bits_per_latent_index_mean: float  # Mean payload normalized by M

    def to_dict(self) -> dict[str, float | int]:
        """Serialize the payload summary into a JSON-safe mapping."""
        return {
            "operational_bits_mean": self.operational_bits_mean,
            "operational_bits_std": self.operational_bits_std,
            "operational_bits_min": self.operational_bits_min,
            "operational_bits_max": self.operational_bits_max,
            "side_information_bits_mean": self.side_information_bits_mean,
            "index_bits_mean": self.index_bits_mean,
            "rate_proxy_bits_mean": self.rate_proxy_bits_mean,
            "bits_per_original_bin_mean": self.bits_per_original_bin_mean,
            "bits_per_reduced_bin_mean": self.bits_per_reduced_bin_mean,
            "bits_per_latent_index_mean": self.bits_per_latent_index_mean,
        }


@dataclass(frozen=True)
class RuntimeCostSummary:
    """Measured host-side deployment latency metrics on the current machine."""

    encode_latency_mean_ms: float  # Mean encode latency per frame [ms]
    encode_latency_std_ms: float  # Standard deviation of encode latency [ms]
    encode_latency_min_ms: float  # Best observed encode latency [ms]
    encode_latency_max_ms: float  # Worst observed encode latency [ms]
    decode_latency_mean_ms: float  # Mean decode latency per frame [ms]
    decode_latency_std_ms: float  # Standard deviation of decode latency [ms]
    decode_latency_min_ms: float  # Best observed decode latency [ms]
    decode_latency_max_ms: float  # Worst observed decode latency [ms]
    roundtrip_exact_fraction: float  # Fraction of timed frames with exact packet round-trip

    def to_dict(self) -> dict[str, float]:
        """Serialize the runtime summary into a JSON-safe mapping."""
        return {
            "encode_latency_mean_ms": self.encode_latency_mean_ms,
            "encode_latency_std_ms": self.encode_latency_std_ms,
            "encode_latency_min_ms": self.encode_latency_min_ms,
            "encode_latency_max_ms": self.encode_latency_max_ms,
            "decode_latency_mean_ms": self.decode_latency_mean_ms,
            "decode_latency_std_ms": self.decode_latency_std_ms,
            "decode_latency_min_ms": self.decode_latency_min_ms,
            "decode_latency_max_ms": self.decode_latency_max_ms,
            "roundtrip_exact_fraction": self.roundtrip_exact_fraction,
        }


@dataclass(frozen=True)
class ModelComplexitySummary:
    """Parameter-count summary for the complete learned demo model."""

    total_parameter_count: int  # All model parameters in the full learned codec
    trainable_parameter_count: int  # Parameters with `requires_grad=True`
    encoder_parameter_count: int  # Encoder-only parameter count
    vector_quantizer_parameter_count: int  # Codebook/VQ parameter count
    decoder_parameter_count: int  # Decoder-only parameter count
    entropy_model_parameter_count: int  # Factorized entropy-model parameter count

    def to_dict(self) -> dict[str, int]:
        """Serialize the complexity summary into a JSON-safe mapping."""
        return {
            "total_parameter_count": self.total_parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "encoder_parameter_count": self.encoder_parameter_count,
            "vector_quantizer_parameter_count": self.vector_quantizer_parameter_count,
            "decoder_parameter_count": self.decoder_parameter_count,
            "entropy_model_parameter_count": self.entropy_model_parameter_count,
        }


@dataclass(frozen=True)
class RateDistortionComplexityReport:
    """Formal deployment-oriented evaluation report for one PSDCodec demo export."""

    export_dir: Path  # Evaluated export root such as `models/exports/demo`
    checkpoint_path: Path  # Checkpoint used to restore the full learned model
    onnx_provider: str  # ONNX Runtime provider used for encoder timing
    validation_reference: ValidationReferenceSummary  # Held-out validation reference metrics
    dataset: EvaluationDatasetSummary  # Dataset provenance and split details
    quality: ReconstructionQualitySummary  # Reconstruction-quality summary
    payload: PayloadCostSummary  # Payload/rate summary
    runtime: RuntimeCostSummary  # Timed encode/decode summary
    complexity: ModelComplexitySummary  # Full-model parameter summary

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report into a JSON-safe mapping."""
        return {
            "export_dir": str(self.export_dir),
            "checkpoint_path": str(self.checkpoint_path),
            "onnx_provider": self.onnx_provider,
            "validation_reference": self.validation_reference.to_dict(),
            "dataset": self.dataset.to_dict(),
            "quality": self.quality.to_dict(),
            "payload": self.payload.to_dict(),
            "runtime": self.runtime.to_dict(),
            "complexity": self.complexity.to_dict(),
        }


@dataclass(frozen=True)
class _ResolvedEvaluationDataset:
    """Internal materialized dataset bundle used by the formal evaluator."""

    dataset: PreparedPsdDataset
    source_kind: Literal["prepared_npz", "raw_campaigns"]
    dataset_path: Path
    evaluation_split: str
    total_frame_count: int
    excluded_campaign_labels: tuple[str, ...] = ()


def demo_eval(
    export_dir: str | Path,  # Export root such as `models/exports/demo`
    *,
    checkpoint_path: str | Path | None = None,  # Optional explicit checkpoint override
    onnx_provider: str = "CPUExecutionProvider",  # ONNX Runtime provider for encode timing
    benchmark_frame_count: int = 64,  # Number of frames materialized for benchmark metrics
    runtime_frame_count: int = 64,  # Number of evaluation frames used for runtime timing
    warmup_frame_count: int = 8,  # Warmup encodes/decodes performed before timing
) -> RateDistortionComplexityReport:
    """Evaluate one demo export in rate-distortion-complexity terms.

    Purpose:
        Produce a deployment-oriented benchmark of the demo by combining the
        export's held-out validation reference metrics with a deterministic
        deployment subset used for operational payload and runtime measurement.
    """
    if benchmark_frame_count <= 0:
        raise ValueError("benchmark_frame_count must be strictly positive.")
    if runtime_frame_count <= 0:
        raise ValueError("runtime_frame_count must be strictly positive.")
    if warmup_frame_count < 0:
        raise ValueError("warmup_frame_count must be non-negative.")

    service, artifacts = create_deployment_service(
        export_dir,
        checkpoint_path=checkpoint_path,
        onnx_provider=onnx_provider,
    )
    validation_reference = _load_validation_reference(artifacts.export_dir)
    resolved_dataset = _load_evaluation_dataset(
        artifacts,
        benchmark_frame_count=benchmark_frame_count,
    )
    evaluation_dataset = resolved_dataset.dataset

    # Evaluate distortion and payload on the deterministic deployment benchmark subset.
    quality = _evaluate_reconstruction_quality(
        service,
        artifacts=artifacts,
        evaluation_dataset=evaluation_dataset,
    )
    payload = _evaluate_payload_cost(
        service,
        evaluation_dataset=evaluation_dataset,
    )

    # Measure encode/decode runtime on a capped, evenly spaced subset so the report
    # stays practical even when the benchmark materialization is larger than the
    # requested timing set.
    runtime_indices = _select_evenly_spaced_indices(
        length=len(evaluation_dataset),
        target_count=runtime_frame_count,
    )
    runtime = _measure_runtime_cost(
        service,
        evaluation_dataset=evaluation_dataset,
        frame_indices=runtime_indices,
        warmup_frame_count=warmup_frame_count,
    )
    complexity = _measure_model_complexity(artifacts.checkpoint_path)

    return RateDistortionComplexityReport(
        export_dir=Path(artifacts.export_dir),
        checkpoint_path=Path(artifacts.checkpoint_path),
        onnx_provider=onnx_provider,
        validation_reference=validation_reference,
        dataset=EvaluationDatasetSummary(
            source_kind=resolved_dataset.source_kind,
            dataset_path=resolved_dataset.dataset_path,
            evaluation_split=resolved_dataset.evaluation_split,
            total_frame_count=resolved_dataset.total_frame_count,
            evaluation_frame_count=len(evaluation_dataset),
            runtime_frame_count=len(runtime_indices),
            original_bin_count=evaluation_dataset.original_bin_count,
            reduced_bin_count=evaluation_dataset.reduced_bin_count,
            block_count=evaluation_dataset.block_count,
            excluded_campaign_labels=resolved_dataset.excluded_campaign_labels,
        ),
        quality=quality,
        payload=payload,
        runtime=runtime,
        complexity=complexity,
    )


def render_rate_distortion_complexity_markdown(
    report: RateDistortionComplexityReport,
) -> str:
    """Render the formal evaluation report as human-readable Markdown."""
    task_line = (
        "n/a"
        if report.quality.task_distortion_mean is None
        else f"{report.quality.task_distortion_mean:.6f}"
    )
    lines = [
        "# PSDCodec Demo Rate-Distortion-Complexity Evaluation",
        "",
        "## Validation Reference",
        "",
        f"- Training summary: `{report.validation_reference.summary_path}`",
        f"- Best epoch: `{report.validation_reference.best_epoch_index}`",
        (
            f"- Held-out mean PSD distortion: "
            f"`{report.validation_reference.psd_distortion_mean:.6f}`"
        ),
        (
            f"- Held-out mean preprocessing-only distortion: "
            f"`{report.validation_reference.preprocessing_distortion_mean:.6f}`"
        ),
        (
            f"- Held-out mean rate proxy: "
            f"`{report.validation_reference.rate_proxy_bits_mean:.3f}` bits/frame"
        ),
        (f"- Held-out task monitor: `{report.validation_reference.task_monitor_mean:.6f}`")
        if report.validation_reference.task_monitor_mean is not None
        else "- Held-out task monitor: `n/a`",
        (f"- Held-out deployment score: `{report.validation_reference.deployment_score:.6f}`")
        if report.validation_reference.deployment_score is not None
        else "- Held-out deployment score: `n/a`",
        "",
        "## Deployment Benchmark",
        "",
        (f"- Export directory: `{report.export_dir}`"),
        f"- Checkpoint: `{report.checkpoint_path}`",
        f"- ONNX provider: `{report.onnx_provider}`",
        f"- Dataset source: `{report.dataset.source_kind}`",
        f"- Dataset path: `{report.dataset.dataset_path}`",
        f"- Benchmark boundary: `{report.dataset.evaluation_split}`",
        f"- Materialized benchmark frames: `{report.dataset.total_frame_count}`",
        f"- Distortion/payload frames: `{report.dataset.evaluation_frame_count}`",
        f"- Timed runtime frames: `{report.dataset.runtime_frame_count}`",
        (
            f"- Frame dimensions: `N={report.dataset.original_bin_count}`, "
            f"`N_r={report.dataset.reduced_bin_count}`, `B={report.dataset.block_count}`"
        ),
        (
            "- Compatibility-excluded campaigns: "
            f"`{', '.join(report.dataset.excluded_campaign_labels)}`"
        )
        if report.dataset.excluded_campaign_labels
        else "- Compatibility-excluded campaigns: `none`",
        "",
        "## Reconstruction Quality",
        "",
        f"- Mean PSD distortion: `{report.quality.psd_distortion_mean:.6f}`",
        f"- PSD distortion std: `{report.quality.psd_distortion_std:.6f}`",
        (
            f"- PSD distortion range: "
            f"`[{report.quality.psd_distortion_min:.6f}, {report.quality.psd_distortion_max:.6f}]`"
        ),
        (
            f"- Mean preprocessing-only distortion: "
            f"`{report.quality.preprocessing_distortion_mean:.6f}`"
        ),
        f"- Mean codec-only distortion: `{report.quality.codec_distortion_mean:.6f}`",
        f"- Mean illustrative task distortion: `{task_line}`",
        "",
        "## Operational Cost",
        "",
        f"- Mean operational payload: `{report.payload.operational_bits_mean:.3f}` bits/frame",
        f"- Payload std: `{report.payload.operational_bits_std:.3f}` bits/frame",
        (
            f"- Payload range: "
            f"`[{report.payload.operational_bits_min}, {report.payload.operational_bits_max}]` bits"
        ),
        (
            f"- Mean side-information payload: "
            f"`{report.payload.side_information_bits_mean:.3f}` bits/frame"
        ),
        f"- Mean index payload: `{report.payload.index_bits_mean:.3f}` bits/frame",
        f"- Mean rate proxy: `{report.payload.rate_proxy_bits_mean:.3f}` bits/frame",
        (f"- Mean bits/original bin: `{report.payload.bits_per_original_bin_mean:.6f}`"),
        f"- Mean bits/reduced bin: `{report.payload.bits_per_reduced_bin_mean:.6f}`",
        (f"- Mean bits/latent index: `{report.payload.bits_per_latent_index_mean:.6f}`"),
        "",
        "## Runtime",
        "",
        (
            f"- Encode latency: `{report.runtime.encode_latency_mean_ms:.3f} ± "
            f"{report.runtime.encode_latency_std_ms:.3f}` ms/frame"
        ),
        (
            f"- Encode latency range: "
            f"`[{report.runtime.encode_latency_min_ms:.3f}, "
            f"{report.runtime.encode_latency_max_ms:.3f}]` ms"
        ),
        (
            f"- Decode latency: `{report.runtime.decode_latency_mean_ms:.3f} ± "
            f"{report.runtime.decode_latency_std_ms:.3f}` ms/frame"
        ),
        (
            f"- Decode latency range: "
            f"`[{report.runtime.decode_latency_min_ms:.3f}, "
            f"{report.runtime.decode_latency_max_ms:.3f}]` ms"
        ),
        (
            f"- Exact packet round-trip fraction on timed frames: "
            f"`{report.runtime.roundtrip_exact_fraction:.6f}`"
        ),
        "",
        "## Model Complexity",
        "",
        f"- Total parameters: `{report.complexity.total_parameter_count}`",
        f"- Trainable parameters: `{report.complexity.trainable_parameter_count}`",
        f"- Encoder parameters: `{report.complexity.encoder_parameter_count}`",
        (f"- Vector-quantizer parameters: `{report.complexity.vector_quantizer_parameter_count}`"),
        f"- Decoder parameters: `{report.complexity.decoder_parameter_count}`",
        (f"- Entropy-model parameters: `{report.complexity.entropy_model_parameter_count}`"),
        "",
        "## Interpretation",
        "",
        (
            "This report is the deployment-oriented rate-distortion-complexity "
            "characterization of the PSDCodec demo: the validation reference keeps "
            "the original held-out training metrics visible, the deployment "
            "benchmark measures practical payload and host-side runtime on a "
            "deterministic raw-frame subset, and the complexity section reports "
            "the size of the complete learned model."
        ),
    ]
    return "\n".join(lines) + "\n"


def _load_evaluation_dataset(
    artifacts: DeploymentArtifacts,
    *,
    benchmark_frame_count: int,
) -> _ResolvedEvaluationDataset:
    """Resolve and materialize the benchmark dataset used for deployment evaluation.

    Strategy:
        1. Prefer the resolved prepared dataset recorded in the export bundle.
        2. If that path is stale, missing, or a Git LFS pointer, fall back to the
           campaign-backed source config when available.
        3. Cap the materialized benchmark to a deterministic deployment-sized subset
           so the formal report remains practical to regenerate inside the repo.
    """
    source_config = _load_source_experiment_config_if_present(artifacts)
    preprocessor = FramePreprocessor(artifacts.runtime_config.preprocessing)

    prepared_dataset = _try_load_prepared_dataset(
        artifacts,
        preprocessor=preprocessor,
    )
    dataset_source_kind: Literal["prepared_npz", "raw_campaigns"]
    dataset_path: Path
    excluded_campaign_labels: tuple[str, ...] = ()
    if prepared_dataset is not None:
        dataset = _subset_dataset_for_benchmark(
            prepared_dataset,
            benchmark_frame_count=benchmark_frame_count,
        )
        dataset_source_kind = "prepared_npz"
        dataset_path = _resolve_exported_dataset_path(
            artifacts.export_dir,
            artifacts.experiment_config.dataset.dataset_path,
        )
    else:
        campaign_config = (
            artifacts.experiment_config
            if artifacts.experiment_config.dataset.source_format == "campaigns"
            else source_config
        )
        if campaign_config is None or campaign_config.dataset.source_format != "campaigns":
            raise FileNotFoundError(
                "Unable to resolve a usable prepared dataset or a campaign-backed source config "
                "for formal demo evaluation.",
            )
        dataset_path = _resolve_exported_dataset_path(
            artifacts.export_dir,
            campaign_config.dataset.dataset_path,
        )
        dataset, excluded_campaign_labels = _load_campaign_dataset_for_evaluation(
            dataset_path,
            preprocessor=preprocessor,
            campaign_config=campaign_config,
            target_bin_count=campaign_config.dataset.campaign_target_bin_count,
            value_scale=campaign_config.dataset.campaign_value_scale,
            max_frames=min(
                benchmark_frame_count,
                campaign_config.dataset.campaign_max_frames or benchmark_frame_count,
            ),
            noise_floor_window=campaign_config.dataset.noise_floor_window,
            noise_floor_percentile=campaign_config.dataset.noise_floor_percentile,
        )
        dataset_source_kind = "raw_campaigns"

    total_frame_count = len(dataset)
    return _ResolvedEvaluationDataset(
        dataset=dataset,
        source_kind=dataset_source_kind,
        dataset_path=dataset_path,
        evaluation_split="deployment_benchmark_subset",
        total_frame_count=total_frame_count,
        excluded_campaign_labels=excluded_campaign_labels,
    )


def _subset_dataset_for_benchmark(
    dataset: PreparedPsdDataset,
    *,
    benchmark_frame_count: int,
) -> PreparedPsdDataset:
    """Return the deterministic leading subset used for the deployment benchmark."""
    if len(dataset) <= benchmark_frame_count:
        return dataset
    benchmark_indices = np.arange(benchmark_frame_count, dtype=np.int64)
    return dataset.subset(benchmark_indices)


def _load_validation_reference(
    export_dir: Path,  # Evaluated export root containing `training_summary.json`
) -> ValidationReferenceSummary:
    """Load held-out validation reference metrics from the exported training summary."""
    summary_path = export_dir / "training_summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    history = summary_payload.get("history", [])
    if not isinstance(history, list) or not history:
        raise ValueError("training_summary.json must contain a non-empty history list.")

    best_epoch_index = int(summary_payload["best_epoch_index"])
    best_epoch_payload = next(
        (
            epoch_payload
            for epoch_payload in history
            if int(epoch_payload["epoch_index"]) == best_epoch_index
        ),
        history[-1],
    )
    return ValidationReferenceSummary(
        summary_path=summary_path,
        best_epoch_index=best_epoch_index,
        psd_distortion_mean=float(best_epoch_payload["validation_psd_loss"]),
        preprocessing_distortion_mean=float(
            best_epoch_payload["validation_preprocessing_psd_loss"]
        ),
        rate_proxy_bits_mean=float(best_epoch_payload["validation_rate_bits"]),
        task_monitor_mean=(
            None
            if "validation_task_monitor" not in best_epoch_payload
            else float(best_epoch_payload["validation_task_monitor"])
        ),
        deployment_score=(
            None
            if "validation_deployment_score" not in best_epoch_payload
            else float(best_epoch_payload["validation_deployment_score"])
        ),
    )


def _load_campaign_dataset_for_evaluation(
    campaign_root: Path,  # Raw campaign root reconstructed from the export metadata
    *,
    preprocessor: FramePreprocessor,
    campaign_config: TrainingExperimentConfig,
    target_bin_count: int | None,
    value_scale: str,
    max_frames: int | None,
    noise_floor_window: int | None,
    noise_floor_percentile: float,
) -> tuple[PreparedPsdDataset, tuple[str, ...]]:
    """Load a campaign-backed evaluation dataset with a support-compatibility fallback.

    Purpose:
        The exported demo cache may be unavailable locally because generated `.npz`
        files are intentionally untracked. When the raw campaign root now contains a
        small number of incompatible campaigns, the evaluator recovers the dominant
        compatible support group and records the dropped campaign labels in the report
        instead of failing with an opaque harmonization error.
    """
    dataset_config = campaign_config.dataset
    try:
        return (
            PreparedPsdDataset.from_campaigns(
                campaign_root,
                preprocessor=preprocessor,
                include_campaign_globs=dataset_config.campaign_include_globs,
                exclude_campaign_globs=dataset_config.campaign_exclude_globs,
                include_node_globs=dataset_config.campaign_node_globs,
                target_bin_count=target_bin_count,
                value_scale=value_scale,
                max_frames=max_frames,
                noise_floor_window=noise_floor_window,
                noise_floor_percentile=noise_floor_percentile,
            ),
            (),
        )
    except CodecConfigurationError as error:
        if "common frequency support" not in str(error):
            raise

    compatible_campaign_labels, excluded_campaign_labels = _select_support_compatible_campaigns(
        campaign_root=campaign_root,
        include_campaign_globs=dataset_config.campaign_include_globs,
        exclude_campaign_globs=dataset_config.campaign_exclude_globs,
        include_node_globs=dataset_config.campaign_node_globs,
    )
    return (
        PreparedPsdDataset.from_campaigns(
            campaign_root,
            preprocessor=preprocessor,
            include_campaign_globs=list(compatible_campaign_labels),
            exclude_campaign_globs=[],
            include_node_globs=dataset_config.campaign_node_globs,
            target_bin_count=target_bin_count,
            value_scale=value_scale,
            max_frames=max_frames,
            noise_floor_window=noise_floor_window,
            noise_floor_percentile=noise_floor_percentile,
        ),
        excluded_campaign_labels,
    )


def _select_support_compatible_campaigns(
    *,
    campaign_root: Path,  # Raw campaign root potentially mixing multiple supports
    include_campaign_globs: list[str] | tuple[str, ...],
    exclude_campaign_globs: list[str] | tuple[str, ...],
    include_node_globs: list[str] | tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the dominant support-compatible campaign subset and dropped labels.

    Raises:
        CodecConfigurationError: If no campaigns match the configured globs or if
            multiple support groups tie for the dominant position.
    """
    support_groups: dict[tuple[float, float], list[str]] = {}
    for campaign_dir in sorted(path for path in campaign_root.iterdir() if path.is_dir()):
        if not _matches_any_glob(campaign_dir.name, include_campaign_globs):
            continue
        if _matches_any_glob(campaign_dir.name, exclude_campaign_globs):
            continue

        support_bounds_hz = _read_campaign_frequency_support_hz(
            campaign_dir,
            include_node_globs=include_node_globs,
        )
        if support_bounds_hz is None:
            continue
        support_groups.setdefault(support_bounds_hz, []).append(campaign_dir.name)

    if not support_groups:
        raise CodecConfigurationError(
            "No campaign directories matched the configured evaluation dataset selection.",
        )

    sorted_groups = sorted(
        (
            (support_bounds_hz, tuple(sorted(labels)))
            for support_bounds_hz, labels in support_groups.items()
        ),
        key=lambda item: (-len(item[1]), item[0]),
    )
    if len(sorted_groups) > 1 and len(sorted_groups[0][1]) == len(sorted_groups[1][1]):
        raise CodecConfigurationError(
            "Campaign support fallback found multiple equally large support groups. "
            "Set explicit campaign globs for evaluation.",
        )

    compatible_campaign_labels = sorted_groups[0][1]
    excluded_campaign_labels = tuple(
        sorted(campaign_label for _, labels in sorted_groups[1:] for campaign_label in labels)
    )
    return compatible_campaign_labels, excluded_campaign_labels


def _read_campaign_frequency_support_hz(
    campaign_dir: Path,  # One raw campaign directory
    *,
    include_node_globs: list[str] | tuple[str, ...],
) -> tuple[float, float] | None:
    """Read the `(start_freq_hz, end_freq_hz)` pair from the first matching node row."""
    node_paths = sorted(
        path
        for path in campaign_dir.iterdir()
        if path.is_file() and _matches_any_glob(path.name, include_node_globs)
    )
    if not node_paths:
        return None

    with node_paths[0].open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        first_row = next(reader, None)
    if first_row is None:
        return None
    return float(first_row["start_freq_hz"]), float(first_row["end_freq_hz"])


def _matches_any_glob(
    value: str,  # Candidate campaign or node label
    patterns: list[str] | tuple[str, ...],
) -> bool:
    """Return whether one string matches at least one configured glob pattern."""
    return any(fnmatch(value, pattern) for pattern in patterns)


def _try_load_prepared_dataset(
    artifacts: DeploymentArtifacts,
    *,
    preprocessor: FramePreprocessor,
) -> PreparedPsdDataset | None:
    """Load the prepared dataset recorded in the export bundle when usable."""
    dataset_config = artifacts.experiment_config.dataset
    if dataset_config.source_format != "npz":
        return None
    dataset_path = _resolve_exported_dataset_path(
        artifacts.export_dir,
        dataset_config.dataset_path,
    )
    if _is_git_lfs_pointer(dataset_path):
        return None

    # Prepared `.npz` archives may be absent or stale when exports move between
    # machines. Failures here trigger a source-config fallback rather than a hard
    # stop because the raw campaigns are the authoritative source.
    try:
        return PreparedPsdDataset.from_npz(
            dataset_path,
            preprocessor=preprocessor,
            frames_key=dataset_config.frames_key,
            frequency_grid_key=dataset_config.frequency_grid_key,
            noise_floor_key=dataset_config.noise_floor_key,
            noise_floor_window=dataset_config.noise_floor_window,
            noise_floor_percentile=dataset_config.noise_floor_percentile,
        )
    except (OSError, ValueError):
        return None


def _measure_model_complexity(
    checkpoint_path: Path,  # Restored checkpoint for the complete learned model
) -> ModelComplexitySummary:
    """Count parameters of the full learned codec restored from a checkpoint."""
    loaded_checkpoint = load_training_checkpoint(checkpoint_path)
    model = TorchFullCodec(loaded_checkpoint.experiment_config.model)
    model.load_state_dict(loaded_checkpoint.model_state_dict)

    def count_parameters(module: Any) -> int:
        """Return the parameter count of one module."""
        return int(sum(int(parameter.numel()) for parameter in module.parameters()))

    total_parameter_count = count_parameters(model)
    trainable_parameter_count = int(
        sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    )
    return ModelComplexitySummary(
        total_parameter_count=total_parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        encoder_parameter_count=count_parameters(model.encoder),
        vector_quantizer_parameter_count=count_parameters(model.vector_quantizer),
        decoder_parameter_count=count_parameters(model.decoder),
        entropy_model_parameter_count=count_parameters(model.entropy_model),
    )


def _evaluate_reconstruction_quality(
    service: PsdCodecService,
    *,
    artifacts: DeploymentArtifacts,
    evaluation_dataset: PreparedPsdDataset,
) -> ReconstructionQualitySummary:
    """Aggregate reconstruction-quality metrics over one evaluation dataset."""
    if evaluation_dataset.frequency_grid_hz is None:
        raise ValueError("evaluation_dataset.frequency_grid_hz is required for formal evaluation.")

    psd_distortions: list[float] = []
    preprocessing_distortions: list[float] = []
    codec_distortions: list[float] = []
    task_distortions: list[float] = []

    for sample_index in range(len(evaluation_dataset)):
        sample = evaluation_dataset[sample_index]
        evaluation = service.evaluate_frame(
            sample.original_frame,
            noise_floor=sample.noise_floor,
            frequency_grid_hz=evaluation_dataset.frequency_grid_hz,
            task_config=artifacts.experiment_config.task,
        )
        psd_distortions.append(evaluation.distortion.psd_distortion)
        preprocessing_distortions.append(evaluation.distortion.preprocessing_distortion)
        codec_distortions.append(evaluation.distortion.codec_distortion)
        if evaluation.distortion.task_distortion is not None:
            task_distortions.append(evaluation.distortion.task_distortion)

    psd_array = np.asarray(psd_distortions, dtype=np.float64)
    return ReconstructionQualitySummary(
        psd_distortion_mean=float(np.mean(psd_array)),
        psd_distortion_std=float(np.std(psd_array)),
        psd_distortion_min=float(np.min(psd_array)),
        psd_distortion_max=float(np.max(psd_array)),
        preprocessing_distortion_mean=float(
            np.mean(np.asarray(preprocessing_distortions, dtype=np.float64))
        ),
        codec_distortion_mean=float(np.mean(np.asarray(codec_distortions, dtype=np.float64))),
        task_distortion_mean=(
            None
            if not task_distortions
            else float(np.mean(np.asarray(task_distortions, dtype=np.float64)))
        ),
    )


def _evaluate_payload_cost(
    service: PsdCodecService,
    *,
    evaluation_dataset: PreparedPsdDataset,
) -> PayloadCostSummary:
    """Aggregate payload-oriented metrics over one evaluation dataset."""
    operational_bits: list[int] = []
    side_information_bits: list[int] = []
    index_bits: list[int] = []
    rate_proxy_bits: list[float] = []

    for sample_index in range(len(evaluation_dataset)):
        sample = evaluation_dataset[sample_index]
        encode_result = service.encode_frame(sample.original_frame)
        operational_bits.append(encode_result.operational_bit_count)
        side_information_bits.append(encode_result.packet.side_information_bit_count)
        index_bits.append(encode_result.packet.index_bit_count)
        rate_proxy_bits.append(encode_result.rate_proxy_bit_count)

    operational_bits_array = np.asarray(operational_bits, dtype=np.float64)
    side_information_bits_array = np.asarray(side_information_bits, dtype=np.float64)
    index_bits_array = np.asarray(index_bits, dtype=np.float64)
    rate_proxy_bits_array = np.asarray(rate_proxy_bits, dtype=np.float64)
    latent_vector_count = int(service.runtime.model.latent_vector_count)
    if latent_vector_count <= 0:
        raise ValueError("service.runtime.model.latent_vector_count must be positive.")
    return PayloadCostSummary(
        operational_bits_mean=float(np.mean(operational_bits_array)),
        operational_bits_std=float(np.std(operational_bits_array)),
        operational_bits_min=int(np.min(operational_bits_array)),
        operational_bits_max=int(np.max(operational_bits_array)),
        side_information_bits_mean=float(np.mean(side_information_bits_array)),
        index_bits_mean=float(np.mean(index_bits_array)),
        rate_proxy_bits_mean=float(np.mean(rate_proxy_bits_array)),
        bits_per_original_bin_mean=float(
            np.mean(operational_bits_array / evaluation_dataset.original_bin_count)
        ),
        bits_per_reduced_bin_mean=float(
            np.mean(operational_bits_array / evaluation_dataset.reduced_bin_count)
        ),
        bits_per_latent_index_mean=float(np.mean(operational_bits_array / latent_vector_count)),
    )


def _measure_runtime_cost(
    service: PsdCodecService,
    *,
    evaluation_dataset: PreparedPsdDataset,
    frame_indices: np.ndarray,
    warmup_frame_count: int,
) -> RuntimeCostSummary:
    """Measure encode/decode latency on a capped subset of evaluation frames."""
    encode_latencies_ms: list[float] = []
    decode_latencies_ms: list[float] = []
    roundtrip_equal_count = 0

    # Warm up the ONNX Runtime session and Python-side codec path before timing.
    for warmup_index in frame_indices[:warmup_frame_count].tolist():
        warmup_frame = evaluation_dataset[warmup_index].original_frame
        warmup_result = service.encode_frame(warmup_frame)
        _ = service.decode_packet(warmup_result.packet_bytes)

    for frame_index in frame_indices.tolist():
        frame = evaluation_dataset[frame_index].original_frame

        encode_start_ns = perf_counter_ns()
        encode_result = service.encode_frame(frame)
        encode_stop_ns = perf_counter_ns()

        decode_start_ns = perf_counter_ns()
        decode_result = service.decode_packet(encode_result.packet_bytes)
        decode_stop_ns = perf_counter_ns()

        encode_latencies_ms.append((encode_stop_ns - encode_start_ns) / 1.0e6)
        decode_latencies_ms.append((decode_stop_ns - decode_start_ns) / 1.0e6)
        if np.allclose(decode_result.reconstructed_frame, encode_result.reconstructed_frame):
            roundtrip_equal_count += 1

    encode_array = np.asarray(encode_latencies_ms, dtype=np.float64)
    decode_array = np.asarray(decode_latencies_ms, dtype=np.float64)
    return RuntimeCostSummary(
        encode_latency_mean_ms=float(np.mean(encode_array)),
        encode_latency_std_ms=float(np.std(encode_array)),
        encode_latency_min_ms=float(np.min(encode_array)),
        encode_latency_max_ms=float(np.max(encode_array)),
        decode_latency_mean_ms=float(np.mean(decode_array)),
        decode_latency_std_ms=float(np.std(decode_array)),
        decode_latency_min_ms=float(np.min(decode_array)),
        decode_latency_max_ms=float(np.max(decode_array)),
        roundtrip_exact_fraction=float(roundtrip_equal_count / len(frame_indices)),
    )


def _select_evenly_spaced_indices(
    *,
    length: int,  # Total number of available evaluation frames
    target_count: int,  # Requested number of timed runtime frames
) -> np.ndarray:
    """Select deterministic runtime-timing indices spanning the evaluation split."""
    if length <= 0:
        raise ValueError("length must be strictly positive.")
    resolved_count = min(length, target_count)
    return np.unique(np.linspace(0, length - 1, num=resolved_count, dtype=np.int64))


def _load_source_experiment_config_if_present(
    artifacts: DeploymentArtifacts,
) -> TrainingExperimentConfig | None:
    """Load the optional source-config sidecar preserved beside one export bundle."""
    experiment_name = artifacts.experiment_config.artifacts.experiment_name
    candidate_paths = [
        artifacts.export_dir / f"{experiment_name}.source.yaml",
        artifacts.export_dir / f"{experiment_name}.source.yml",
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return TrainingExperimentConfig.from_yaml(candidate_path)
    return None


def _resolve_exported_dataset_path(
    export_dir: Path,  # Export root used to infer the repository root
    candidate_path: str | Path,  # Dataset path recorded in the export metadata
) -> Path:
    """Resolve a dataset path recorded inside one export bundle.

    Purpose:
        Export metadata may store either repository-relative paths or absolute paths
        from another workstation. This helper resolves relative paths against the
        current repository root and provides a small prepared-dataset fallback for
        stale absolute cache locations.
    """
    candidate = Path(candidate_path)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        repo_root = export_dir.resolve().parents[2]
        prepared_fallback = repo_root / "data" / "processed" / candidate.name
        if prepared_fallback.exists():
            return prepared_fallback
        return candidate

    repo_root = export_dir.resolve().parents[2]
    return repo_root / candidate


def _is_git_lfs_pointer(
    path: Path,  # Candidate file that might be a Git LFS text pointer
) -> bool:
    """Return whether a file looks like a Git LFS pointer rather than real data."""
    if not path.exists() or not path.is_file():
        return False
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        first_line = stream.readline().strip()
    return first_line == "version https://git-lfs.github.com/spec/v1"
