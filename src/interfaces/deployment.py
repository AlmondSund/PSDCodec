"""Deployment-facing helpers for loading, evaluating, and reporting demo exports."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from codec.config import (
    CodecRuntimeConfig,
    FactorizedEntropyModelConfig,
    PacketFormatConfig,
    PreprocessingConfig,
    ScalarQuantizerConfig,
)
from codec.exceptions import CodecConfigurationError
from data.campaigns import CampaignDatasetBundle, load_campaign_dataset_bundle
from interfaces.api import PsdCodecService
from models.torch_backend import TorchFullCodec
from objectives.distortion import IllustrativeTaskConfig
from pipelines.runtime import CodecEvaluation
from pipelines.training import TrainingExperimentConfig, load_training_checkpoint
from utils import FloatArray, as_1d_float_array

try:
    import onnxruntime as _ort  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised only when onnxruntime is unavailable
    _ort = cast(Any, None)

try:
    import torch as _torch
except ImportError:  # pragma: no cover - exercised only when torch is unavailable
    _torch = cast(Any, None)


@dataclass(frozen=True)
class DeploymentArtifacts:
    """Deployment assets loaded from one trained experiment export directory."""

    export_dir: Path  # Experiment export root such as `models/exports/<name>`
    runtime_asset_dir: Path  # Directory containing runtime JSON and NumPy assets
    onnx_path: Path  # Encoder-only ONNX export used on the deployment node
    checkpoint_path: Path  # Server-side checkpoint used to restore the decoder
    runtime_config: CodecRuntimeConfig  # Exported deterministic codec configuration
    experiment_config: TrainingExperimentConfig  # Full saved training experiment configuration
    codebook: FloatArray  # Runtime VQ codebook with shape [J, d]
    probabilities: FloatArray | None = None  # Optional fitted entropy-model PMF

    @property
    def original_bin_count(self) -> int:
        """Return the harmonized PSD length expected by the trained deployment assets."""
        dataset_config = self.experiment_config.dataset
        if dataset_config.source_format == "campaigns":
            if dataset_config.campaign_target_bin_count is None:
                raise CodecConfigurationError(
                    "Campaign-backed deployment assets require campaign_target_bin_count.",
                )
            return dataset_config.campaign_target_bin_count
        raise CodecConfigurationError(
            "original_bin_count is only defined for campaign-backed deployment assets.",
        )


@dataclass(frozen=True)
class CampaignFrameSample:
    """One PSD frame sample loaded from the raw campaign corpus for deployment demos."""

    frame: FloatArray  # Original harmonized PSD frame in linear power
    frequency_grid_hz: FloatArray  # Shared frequency support [Hz]
    noise_floor: FloatArray | None  # Optional sequence-local reference baseline
    campaign_label: str  # Campaign directory label
    node_label: str  # Node CSV stem
    sequence_id: str  # Stable sequence identifier `campaign/node`
    timestamp_ms: int  # Acquisition timestamp [ms since epoch]


@dataclass(frozen=True)
class DeploymentFrameReport:
    """Full deployment evaluation payload for one campaign frame."""

    frame_index: int  # Index inside the deterministic campaign ordering
    campaign_label: str  # Campaign directory label
    node_label: str  # Node CSV stem
    sequence_id: str  # Stable `campaign/node` identifier
    timestamp_ms: int  # Acquisition timestamp [ms since epoch]
    frequency_grid_hz: FloatArray  # Shared frequency support [Hz]
    original_frame: FloatArray  # Original harmonized PSD frame
    preprocessing_only_frame: FloatArray  # Deterministic preprocessing-only reconstruction
    reconstructed_frame: FloatArray  # Full deployment reconstruction
    noise_floor: FloatArray | None  # Optional sequence-local baseline
    operational_bit_count: int  # Packet size in operational bits
    rate_proxy_bit_count: float  # Training-time rate proxy for the same frame
    side_information_bit_count: int  # Side-information contribution [bits]
    index_bit_count: int  # Latent-index payload contribution [bits]
    psd_distortion: float  # D_psd(original, reconstructed)
    preprocessing_distortion: float  # D_psd(original, preprocessing_only)
    codec_distortion: float  # D_psd(preprocessing_only, reconstructed)
    peak_frequency_error_hz: float  # Absolute dominant-peak location error [Hz]
    peak_power_error_db: float  # Absolute dominant-peak amplitude error [dB]
    roundtrip_equal: bool  # Whether packet decode matches the encode-time reconstruction exactly
    task_distortion: float | None = None  # Optional illustrative task distortion


@dataclass(frozen=True)
class DeploymentBatchSummary:
    """Aggregate metrics over a batch of deployment frame evaluations."""

    frame_count: int  # Number of evaluated frames
    all_roundtrip_equal: bool  # Packet decode matches encode-time reconstruction for every frame
    packet_bits_mean: float  # Mean operational packet size [bits]
    packet_bits_std: float  # Standard deviation of packet size [bits]
    packet_bits_min: int  # Minimum operational packet size [bits]
    packet_bits_max: int  # Maximum operational packet size [bits]
    rate_proxy_bits_mean: float  # Mean training-time rate proxy [bits]
    rate_proxy_bits_std: float  # Standard deviation of the rate proxy [bits]
    psd_distortion_mean: float  # Mean D_psd(original, reconstructed)
    psd_distortion_std: float  # Standard deviation of D_psd
    psd_distortion_min: float  # Minimum D_psd across the batch
    psd_distortion_max: float  # Maximum D_psd across the batch
    preprocessing_distortion_mean: float  # Mean preprocessing-only distortion
    codec_distortion_mean: float  # Mean codec-only distortion
    peak_frequency_error_hz_mean: float  # Mean dominant-peak location error [Hz]
    peak_frequency_error_hz_max: float  # Worst dominant-peak location error [Hz]
    peak_power_error_db_mean: float  # Mean dominant-peak amplitude error [dB]
    peak_power_error_db_max: float  # Worst dominant-peak amplitude error [dB]
    task_distortion_mean: float | None = None  # Optional mean illustrative task loss

    def to_display_dict(self) -> dict[str, float | int | bool | None]:
        """Return a notebook-friendly dictionary representation of the summary."""
        return {
            "frame_count": self.frame_count,
            "all_roundtrip_equal": self.all_roundtrip_equal,
            "packet_bits_mean": self.packet_bits_mean,
            "packet_bits_std": self.packet_bits_std,
            "packet_bits_min": self.packet_bits_min,
            "packet_bits_max": self.packet_bits_max,
            "rate_proxy_bits_mean": self.rate_proxy_bits_mean,
            "rate_proxy_bits_std": self.rate_proxy_bits_std,
            "psd_distortion_mean": self.psd_distortion_mean,
            "psd_distortion_std": self.psd_distortion_std,
            "psd_distortion_min": self.psd_distortion_min,
            "psd_distortion_max": self.psd_distortion_max,
            "preprocessing_distortion_mean": self.preprocessing_distortion_mean,
            "codec_distortion_mean": self.codec_distortion_mean,
            "peak_frequency_error_khz_mean": self.peak_frequency_error_hz_mean / 1.0e3,
            "peak_frequency_error_khz_max": self.peak_frequency_error_hz_max / 1.0e3,
            "peak_power_error_db_mean": self.peak_power_error_db_mean,
            "peak_power_error_db_max": self.peak_power_error_db_max,
            "task_distortion_mean": self.task_distortion_mean,
        }


@dataclass(frozen=True)
class DeploymentReadinessAssessment:
    """Heuristic notebook-scale judgment of deployment readiness."""

    verdict: Literal["deployment_good", "borderline", "undertrained"]  # Qualitative verdict
    reasons: tuple[str, ...]  # Human-readable evidence supporting the verdict


@dataclass(frozen=True)
class DeploymentBatchReport:
    """Combined per-frame and aggregate deployment-analysis output."""

    frame_reports: tuple[DeploymentFrameReport, ...]  # Per-frame evaluation payloads
    summary: DeploymentBatchSummary  # Aggregate metrics over the batch
    assessment: DeploymentReadinessAssessment  # Qualitative readiness triage


@dataclass
class OnnxTorchDeploymentModel:
    """Deployment adapter using ONNX Runtime for encode and PyTorch for decode.

    The sensing-node boundary is the exported encoder ONNX graph. The cloud-side
    reconstruction still needs the trained decoder weights, so this adapter keeps
    the decoder in PyTorch while routing the encode path through ONNX Runtime.
    """

    session: Any  # `onnxruntime.InferenceSession`
    decoder: Any  # Restored PyTorch decoder module
    reduced_bin_count: int  # Preprocessed input length N_r
    latent_vector_count: int  # Number of latent positions M
    embedding_dim: int  # Latent embedding dimension d
    input_name: str  # ONNX encoder input tensor name
    output_name: str  # ONNX encoder output tensor name

    def encode(
        self,
        normalized_frame: FloatArray,  # Standardized one-dimensional frame u_t
    ) -> FloatArray:
        """Encode one normalized frame through the exported ONNX encoder."""
        frame = as_1d_float_array(
            normalized_frame,
            name="normalized_frame",
            allow_negative=True,
        )
        if frame.size != self.reduced_bin_count:
            raise CodecConfigurationError(
                "normalized_frame length does not match the ONNX encoder input size.",
            )

        # The deployment boundary runs on float32 because the ONNX export was traced that way.
        encoder_input = frame.astype(np.float32, copy=False)[None, :]
        latent_batch = np.asarray(
            self.session.run([self.output_name], {self.input_name: encoder_input})[0],
            dtype=np.float64,
        )
        expected_shape = (1, self.latent_vector_count, self.embedding_dim)
        if latent_batch.shape != expected_shape:
            raise CodecConfigurationError(
                f"ONNX encoder returned shape {latent_batch.shape}, expected {expected_shape}."
            )
        return cast(FloatArray, latent_batch[0])

    def decode(
        self,
        quantized_latents: FloatArray,  # Quantized latent matrix with shape [M, d]
    ) -> FloatArray:
        """Decode one quantized latent matrix with the restored PyTorch decoder."""
        latent_matrix = np.asarray(quantized_latents, dtype=np.float64)
        expected_shape = (self.latent_vector_count, self.embedding_dim)
        if latent_matrix.shape != expected_shape:
            raise CodecConfigurationError(
                f"quantized_latents has shape {latent_matrix.shape}, expected {expected_shape}."
            )
        if _torch is None:
            raise ImportError("PyTorch is required to decode latents on the server side.")

        # The server-side decoder is restored from the best training checkpoint.
        with _torch.no_grad():
            latent_tensor = _torch.as_tensor(latent_matrix[None, :, :], dtype=_torch.float32)
            decoded = self.decoder(latent_tensor).cpu().numpy()
        return np.asarray(decoded[0], dtype=np.float64)


def create_deployment_service(
    export_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    onnx_provider: str = "CPUExecutionProvider",
) -> tuple[PsdCodecService, DeploymentArtifacts]:
    """Create an operational deployment service from exported artifacts.

    Args:
        export_dir: Experiment export directory such as `models/exports/demo`.
        checkpoint_path: Optional explicit checkpoint path. When omitted, the function
            uses the best checkpoint recorded in `training_summary.json`.
        onnx_provider: ONNX Runtime provider used for the encoder boundary.

    Returns:
        A ready-to-use `PsdCodecService` plus the loaded deployment asset bundle.
    """
    artifacts = load_deployment_artifacts(export_dir, checkpoint_path=checkpoint_path)
    deployment_model = load_onnx_torch_deployment_model(
        artifacts,
        onnx_provider=onnx_provider,
    )
    service = PsdCodecService.create(
        artifacts.runtime_config,
        model=deployment_model,
        codebook=artifacts.codebook,
        probabilities=artifacts.probabilities,
    )
    return service, artifacts


def load_deployment_artifacts(
    export_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
) -> DeploymentArtifacts:
    """Load exported runtime assets, configuration, and the paired checkpoint."""
    resolved_export_dir = Path(export_dir)
    runtime_asset_dir = resolved_export_dir / "runtime_assets"
    summary_path = resolved_export_dir / "training_summary.json"
    onnx_path = resolved_export_dir / "encoder.onnx"
    if not summary_path.exists():
        raise CodecConfigurationError(f"Missing training summary file: {summary_path}.")
    if not runtime_asset_dir.exists():
        raise CodecConfigurationError(f"Missing runtime asset directory: {runtime_asset_dir}.")
    if not onnx_path.exists():
        raise CodecConfigurationError(f"Missing ONNX encoder export: {onnx_path}.")

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    experiment_config = TrainingExperimentConfig.from_dict(
        cast(dict[str, Any], summary_payload["experiment_config"])
    )
    exported_source_config = _load_exported_source_config_if_present(
        resolved_export_dir,
        experiment_name=experiment_config.artifacts.experiment_name,
    )
    if (
        exported_source_config is not None
        and exported_source_config.to_dict() != experiment_config.to_dict()
    ):
        raise CodecConfigurationError(
            "The exported training_summary.json does not match the copied experiment YAML. "
            "These deployment artifacts are stale relative to the current canonical "
            "configuration and must be regenerated by retraining."
        )
    resolved_checkpoint_path = _resolve_checkpoint_path(
        export_dir=resolved_export_dir,
        summary_payload=summary_payload,
        checkpoint_path=checkpoint_path,
    )
    runtime_config = load_runtime_config_json(runtime_asset_dir / "runtime_config.json")
    codebook = np.asarray(np.load(runtime_asset_dir / "codebook.npy"), dtype=np.float64)
    probabilities = None
    probability_path = runtime_asset_dir / "entropy_probabilities.npy"
    if probability_path.exists():
        probabilities = np.asarray(np.load(probability_path), dtype=np.float64)
    return DeploymentArtifacts(
        export_dir=resolved_export_dir,
        runtime_asset_dir=runtime_asset_dir,
        onnx_path=onnx_path,
        checkpoint_path=resolved_checkpoint_path,
        runtime_config=runtime_config,
        experiment_config=experiment_config,
        codebook=codebook,
        probabilities=probabilities,
    )


def load_onnx_torch_deployment_model(
    artifacts: DeploymentArtifacts,
    *,
    onnx_provider: str = "CPUExecutionProvider",
) -> OnnxTorchDeploymentModel:
    """Load the ONNX encoder session and the paired server-side PyTorch decoder."""
    if _ort is None:
        raise ImportError(
            "onnxruntime is required to run the deployment encoder boundary.",
        )
    if _torch is None:
        raise ImportError("PyTorch is required to restore the server-side decoder.")

    checkpoint = load_training_checkpoint(artifacts.checkpoint_path)
    full_model = TorchFullCodec(checkpoint.experiment_config.model)
    full_model.load_state_dict(checkpoint.model_state_dict)
    full_model.eval()

    # The notebook intentionally pins the provider so the deployment backend is explicit.
    session = _ort.InferenceSession(str(artifacts.onnx_path), providers=[onnx_provider])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return OnnxTorchDeploymentModel(
        session=session,
        decoder=full_model.decoder,
        reduced_bin_count=checkpoint.experiment_config.model.reduced_bin_count,
        latent_vector_count=checkpoint.experiment_config.model.latent_vector_count,
        embedding_dim=checkpoint.experiment_config.model.embedding_dim,
        input_name=input_name,
        output_name=output_name,
    )


def load_campaign_frame_sample(
    artifacts: DeploymentArtifacts,
    *,
    frame_index: int = 0,
) -> CampaignFrameSample:
    """Load one demo PSD frame from the campaign corpus used during training."""
    return load_campaign_frame_samples(
        artifacts,
        frame_indices=[frame_index],
    )[0]


def load_campaign_frame_samples(
    artifacts: DeploymentArtifacts,
    *,
    frame_indices: Sequence[int] | None = None,
    max_frames: int | None = None,
) -> tuple[CampaignFrameSample, ...]:
    """Load multiple campaign-backed PSD frames with one bundle parse.

    Purpose:
        Support notebook-scale deployment analysis over many frames without repeatedly
        reparsing the raw campaign CSV files.

    Args:
        artifacts: Deployment artifact bundle created from one trained export directory.
        frame_indices: Optional explicit frame indices to load, in the requested order.
        max_frames: Number of leading frames to load when `frame_indices` is omitted.

    Returns:
        A tuple of `CampaignFrameSample` objects ordered exactly as requested.

    Raises:
        CodecConfigurationError: If the deployment artifacts are not campaign-backed,
            if the requested frame indices are invalid, or if the campaign bundle does
            not contain enough frames.
    """
    selected_indices = _resolve_requested_frame_indices(
        frame_indices=frame_indices,
        max_frames=max_frames,
    )
    dataset_config = artifacts.experiment_config.dataset
    if dataset_config.source_format != "campaigns":
        raise CodecConfigurationError(
            "load_campaign_frame_samples requires campaign-backed deployment assets.",
        )

    resolved_dataset_path = _resolve_repository_path(
        export_dir=artifacts.export_dir,
        candidate_path=dataset_config.dataset_path,
    )
    bundle = load_campaign_dataset_bundle(
        resolved_dataset_path,
        include_campaign_globs=dataset_config.campaign_include_globs,
        exclude_campaign_globs=dataset_config.campaign_exclude_globs,
        include_node_globs=dataset_config.campaign_node_globs,
        target_bin_count=artifacts.original_bin_count,
        value_scale=dataset_config.campaign_value_scale,
        max_frames=max(selected_indices) + 1,
        noise_floor_window=dataset_config.noise_floor_window,
        noise_floor_percentile=dataset_config.noise_floor_percentile,
    )
    available_frames = bundle.frames.shape[0]
    if max(selected_indices) >= available_frames:
        raise CodecConfigurationError(
            "Requested frame index "
            f"{max(selected_indices)} but only {available_frames} frames were loaded."
        )

    return tuple(
        _build_campaign_frame_sample(bundle, frame_index)
        for frame_index in selected_indices
    )


def load_runtime_config_json(config_path: str | Path) -> CodecRuntimeConfig:
    """Load the exported runtime configuration JSON into typed dataclasses."""
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    preprocessing_payload = cast(dict[str, Any], payload["preprocessing"])
    entropy_payload = cast(dict[str, Any], payload["entropy_model"])
    packet_payload = dict(cast(dict[str, Any], payload.get("packet_format", {})))
    if "magic" in packet_payload:
        packet_payload["magic"] = str(packet_payload["magic"]).encode("ascii")

    mean_quantizer_payload = cast(dict[str, Any], preprocessing_payload["mean_quantizer"])
    log_sigma_quantizer_payload = cast(dict[str, Any], preprocessing_payload["log_sigma_quantizer"])
    preprocessing_fields = dict(preprocessing_payload)
    preprocessing_fields["mean_quantizer"] = ScalarQuantizerConfig(**mean_quantizer_payload)
    preprocessing_fields["log_sigma_quantizer"] = ScalarQuantizerConfig(
        **log_sigma_quantizer_payload,
    )
    return CodecRuntimeConfig(
        preprocessing=PreprocessingConfig(**preprocessing_fields),
        entropy_model=FactorizedEntropyModelConfig(**entropy_payload),
        packet_format=PacketFormatConfig(**packet_payload),
    )


def _resolve_checkpoint_path(
    *,
    export_dir: Path,
    summary_payload: dict[str, Any],
    checkpoint_path: str | Path | None,
) -> Path:
    """Resolve the checkpoint path recorded in the training summary."""
    if checkpoint_path is not None:
        candidate = Path(checkpoint_path)
    else:
        best_checkpoint = summary_payload.get("best_checkpoint_path")
        if best_checkpoint is None:
            raise CodecConfigurationError(
                "training_summary.json does not record a best checkpoint path.",
            )
        candidate = Path(str(best_checkpoint))
    return _resolve_repository_path(
        export_dir=export_dir,
        candidate_path=candidate,
    )


def _resolve_repository_path(
    *,
    export_dir: Path,
    candidate_path: str | Path,
) -> Path:
    """Resolve a repository-relative path recorded in exported experiment metadata."""
    candidate = Path(candidate_path)
    if candidate.is_absolute():
        return candidate

    # Export directories live under `models/exports/<experiment_name>`, so `parents[2]`
    # resolves to the repository root where relative experiment paths are anchored.
    repo_root = export_dir.resolve().parents[2]
    return repo_root / candidate


def _resolve_requested_frame_indices(
    *,
    frame_indices: Sequence[int] | None,
    max_frames: int | None,
) -> tuple[int, ...]:
    """Normalize the requested campaign frame indices for batch loading."""
    if frame_indices is not None:
        normalized = tuple(int(frame_index) for frame_index in frame_indices)
        if not normalized:
            raise CodecConfigurationError("frame_indices must contain at least one index.")
    else:
        resolved_max_frames = 1 if max_frames is None else max_frames
        if resolved_max_frames <= 0:
            raise CodecConfigurationError("max_frames must be strictly positive.")
        normalized = tuple(range(resolved_max_frames))

    if any(frame_index < 0 for frame_index in normalized):
        raise CodecConfigurationError("frame indices must be non-negative.")
    return normalized


def _load_exported_source_config_if_present(
    export_dir: Path,
    *,
    experiment_name: str,
) -> TrainingExperimentConfig | None:
    """Load the copied source YAML from one export directory when available."""
    canonical_path = export_dir / f"{experiment_name}.yaml"
    candidate_paths: list[Path] = []
    if canonical_path.exists():
        candidate_paths.append(canonical_path)
    if not candidate_paths:
        candidate_paths = sorted(export_dir.glob("*.yaml")) + sorted(export_dir.glob("*.yml"))
    if not candidate_paths:
        return None
    return TrainingExperimentConfig.from_yaml(candidate_paths[0])


def _build_campaign_frame_sample(
    bundle: CampaignDatasetBundle,
    frame_index: int,
) -> CampaignFrameSample:
    """Convert one bundle row into the public `CampaignFrameSample` dataclass."""
    noise_floor = None if bundle.noise_floors is None else bundle.noise_floors[frame_index]
    if (
        bundle.campaign_labels is None
        or bundle.node_labels is None
        or bundle.sequence_ids is None
        or bundle.timestamps_ms is None
    ):
        raise CodecConfigurationError(
            "Campaign metadata arrays are missing from the loaded deployment sample bundle.",
        )
    return CampaignFrameSample(
        frame=np.asarray(bundle.frames[frame_index], dtype=np.float64),
        frequency_grid_hz=np.asarray(bundle.frequency_grid_hz, dtype=np.float64),
        noise_floor=None if noise_floor is None else np.asarray(noise_floor, dtype=np.float64),
        campaign_label=str(bundle.campaign_labels[frame_index]),
        node_label=str(bundle.node_labels[frame_index]),
        sequence_id=str(bundle.sequence_ids[frame_index]),
        timestamp_ms=int(bundle.timestamps_ms[frame_index]),
    )


def evaluate_deployment_samples(
    service: PsdCodecService,
    *,
    samples: Sequence[CampaignFrameSample],
    frame_indices: Sequence[int] | None = None,
    task_config: IllustrativeTaskConfig | None = None,
) -> DeploymentBatchReport:
    """Evaluate preloaded campaign samples through the deployed codec."""
    if not samples:
        raise CodecConfigurationError("samples must contain at least one frame.")
    if frame_indices is None:
        resolved_frame_indices = tuple(range(len(samples)))
    else:
        resolved_frame_indices = tuple(int(frame_index) for frame_index in frame_indices)
    if len(resolved_frame_indices) != len(samples):
        raise CodecConfigurationError(
            "frame_indices must have the same length as samples when provided."
        )
    if any(frame_index < 0 for frame_index in resolved_frame_indices):
        raise CodecConfigurationError("frame_indices must be non-negative.")

    return _evaluate_samples(
        service,
        samples=samples,
        frame_indices=resolved_frame_indices,
        task_config=task_config,
    )


def evaluate_deployment_batch(
    service: PsdCodecService,
    artifacts: DeploymentArtifacts,
    *,
    max_frames: int = 24,
    frame_indices: Sequence[int] | None = None,
    task_config: IllustrativeTaskConfig | None = None,
) -> DeploymentBatchReport:
    """Evaluate a batch of campaign frames through the deployed codec."""
    selected_indices = _resolve_requested_frame_indices(
        frame_indices=frame_indices,
        max_frames=max_frames,
    )
    samples = load_campaign_frame_samples(
        artifacts,
        frame_indices=selected_indices,
    )

    return _evaluate_samples(
        service,
        samples=samples,
        frame_indices=selected_indices,
        task_config=task_config,
    )


def select_gallery_frames(
    report: DeploymentBatchReport,
    *,
    gallery_size: int = 6,
) -> tuple[DeploymentFrameReport, ...]:
    """Select representative deployment frames ordered across the distortion range."""
    if gallery_size <= 0:
        raise CodecConfigurationError("gallery_size must be strictly positive.")
    if not report.frame_reports:
        raise CodecConfigurationError("report.frame_reports must contain at least one frame.")

    sorted_reports = sorted(
        report.frame_reports,
        key=lambda frame_report: (
            frame_report.psd_distortion,
            frame_report.operational_bit_count,
            frame_report.frame_index,
        ),
    )
    target_size = min(gallery_size, len(sorted_reports))
    quantile_positions = np.linspace(0, len(sorted_reports) - 1, num=target_size)
    selected_indices: list[int] = []
    for position in quantile_positions:
        candidate_index = int(round(float(position)))
        if candidate_index not in selected_indices:
            selected_indices.append(candidate_index)

    for candidate_index in range(len(sorted_reports)):
        if len(selected_indices) >= target_size:
            break
        if candidate_index not in selected_indices:
            selected_indices.append(candidate_index)

    return tuple(sorted_reports[candidate_index] for candidate_index in selected_indices)


def assess_deployment_readiness(
    summary: DeploymentBatchSummary,
) -> DeploymentReadinessAssessment:
    """Return a conservative heuristic deployment-readiness verdict."""
    reasons: list[str] = []
    severity_scores: list[int] = []

    if not summary.all_roundtrip_equal:
        severity_scores.append(2)
        reasons.append("Packet decode is not deterministic across the evaluated frames.")
    else:
        reasons.append("Packet decode matches the encode-time reconstruction on every frame.")

    if summary.psd_distortion_mean > 0.50:
        severity_scores.append(2)
        reasons.append(
            f"Mean PSD distortion is high ({summary.psd_distortion_mean:.3f}), "
            "which is consistent with visible reconstruction loss."
        )
    elif summary.psd_distortion_mean > 0.30:
        severity_scores.append(1)
        reasons.append(f"Mean PSD distortion is moderate ({summary.psd_distortion_mean:.3f}).")
    else:
        reasons.append(
            f"Mean PSD distortion is low enough for a promising deployment candidate "
            f"({summary.psd_distortion_mean:.3f})."
        )

    if summary.peak_power_error_db_mean > 8.0:
        severity_scores.append(2)
        reasons.append(
            "Mean dominant-peak amplitude error is large "
            f"({summary.peak_power_error_db_mean:.2f} dB)."
        )
    elif summary.peak_power_error_db_mean > 4.0:
        severity_scores.append(1)
        reasons.append(
            "Mean dominant-peak amplitude error is noticeable "
            f"({summary.peak_power_error_db_mean:.2f} dB)."
        )
    else:
        reasons.append(
            f"Dominant-peak amplitude preservation looks acceptable "
            f"({summary.peak_power_error_db_mean:.2f} dB mean error)."
        )

    if summary.peak_frequency_error_hz_mean > 2.0e5:
        severity_scores.append(1)
        reasons.append(
            "Mean dominant-peak frequency error is "
            f"{summary.peak_frequency_error_hz_mean / 1.0e3:.1f} kHz."
        )
    else:
        reasons.append(
            f"Dominant-peak frequency alignment is stable "
            f"({summary.peak_frequency_error_hz_mean / 1.0e3:.1f} kHz mean error)."
        )

    packet_variation_fraction = summary.packet_bits_std / max(summary.packet_bits_mean, 1.0)
    if packet_variation_fraction > 0.10:
        severity_scores.append(1)
        reasons.append(
            f"Packet-size variation is fairly wide ({packet_variation_fraction:.1%} relative std)."
        )
    else:
        reasons.append(
            f"Packet size is operationally stable ({packet_variation_fraction:.1%} relative std)."
        )

    max_severity = max(severity_scores, default=0)
    if max_severity >= 2:
        verdict: Literal["deployment_good", "borderline", "undertrained"] = "undertrained"
    elif max_severity == 1:
        verdict = "borderline"
    else:
        verdict = "deployment_good"
    return DeploymentReadinessAssessment(verdict=verdict, reasons=tuple(reasons))


def _build_frame_report(
    *,
    frame_index: int,
    sample: CampaignFrameSample,
    evaluation: CodecEvaluation,
    roundtrip_equal: bool,
) -> DeploymentFrameReport:
    """Convert one runtime evaluation into a stable frame-level deployment report."""
    peak_frequency_error_hz = _peak_frequency_error_hz(
        sample.frame,
        evaluation.encode_result.reconstructed_frame,
        sample.frequency_grid_hz,
    )
    peak_power_error_db = _peak_power_error_db(
        sample.frame,
        evaluation.encode_result.reconstructed_frame,
    )
    return DeploymentFrameReport(
        frame_index=frame_index,
        campaign_label=sample.campaign_label,
        node_label=sample.node_label,
        sequence_id=sample.sequence_id,
        timestamp_ms=sample.timestamp_ms,
        frequency_grid_hz=np.asarray(sample.frequency_grid_hz, dtype=np.float64),
        original_frame=np.asarray(sample.frame, dtype=np.float64),
        preprocessing_only_frame=np.asarray(
            evaluation.encode_result.preprocessing_only_frame,
            dtype=np.float64,
        ),
        reconstructed_frame=np.asarray(
            evaluation.encode_result.reconstructed_frame,
            dtype=np.float64,
        ),
        noise_floor=(
            None
            if sample.noise_floor is None
            else np.asarray(sample.noise_floor, dtype=np.float64)
        ),
        operational_bit_count=evaluation.encode_result.operational_bit_count,
        rate_proxy_bit_count=evaluation.encode_result.rate_proxy_bit_count,
        side_information_bit_count=evaluation.encode_result.packet.side_information_bit_count,
        index_bit_count=evaluation.encode_result.packet.index_bit_count,
        psd_distortion=evaluation.distortion.psd_distortion,
        preprocessing_distortion=evaluation.distortion.preprocessing_distortion,
        codec_distortion=evaluation.distortion.codec_distortion,
        peak_frequency_error_hz=peak_frequency_error_hz,
        peak_power_error_db=peak_power_error_db,
        roundtrip_equal=roundtrip_equal,
        task_distortion=evaluation.distortion.task_distortion,
    )


def _evaluate_samples(
    service: PsdCodecService,
    *,
    samples: Sequence[CampaignFrameSample],
    frame_indices: Sequence[int],
    task_config: IllustrativeTaskConfig | None,
) -> DeploymentBatchReport:
    """Run frame-wise deployment evaluation over an already-loaded sample collection."""
    frame_reports: list[DeploymentFrameReport] = []
    for frame_index, sample in zip(frame_indices, samples, strict=True):
        evaluation = service.evaluate_frame(
            sample.frame,
            noise_floor=sample.noise_floor,
            frequency_grid_hz=sample.frequency_grid_hz,
            task_config=task_config,
        )
        decoded = service.decode_packet(evaluation.encode_result.packet_bytes)
        frame_reports.append(
            _build_frame_report(
                frame_index=frame_index,
                sample=sample,
                evaluation=evaluation,
                roundtrip_equal=bool(
                    np.allclose(
                        decoded.reconstructed_frame,
                        evaluation.encode_result.reconstructed_frame,
                    )
                ),
            )
        )

    summary = _summarize_frame_reports(frame_reports)
    return DeploymentBatchReport(
        frame_reports=tuple(frame_reports),
        summary=summary,
        assessment=assess_deployment_readiness(summary),
    )


def _summarize_frame_reports(
    frame_reports: Sequence[DeploymentFrameReport],
) -> DeploymentBatchSummary:
    """Aggregate the numeric metrics required for the deployment summary section."""
    if not frame_reports:
        raise CodecConfigurationError("frame_reports must contain at least one frame.")

    packet_bits = np.asarray(
        [frame_report.operational_bit_count for frame_report in frame_reports],
        dtype=np.float64,
    )
    rate_proxy_bits = np.asarray(
        [frame_report.rate_proxy_bit_count for frame_report in frame_reports],
        dtype=np.float64,
    )
    psd_distortions = np.asarray(
        [frame_report.psd_distortion for frame_report in frame_reports],
        dtype=np.float64,
    )
    preprocessing_distortions = np.asarray(
        [frame_report.preprocessing_distortion for frame_report in frame_reports],
        dtype=np.float64,
    )
    codec_distortions = np.asarray(
        [frame_report.codec_distortion for frame_report in frame_reports],
        dtype=np.float64,
    )
    peak_frequency_errors_hz = np.asarray(
        [frame_report.peak_frequency_error_hz for frame_report in frame_reports],
        dtype=np.float64,
    )
    peak_power_errors_db = np.asarray(
        [frame_report.peak_power_error_db for frame_report in frame_reports],
        dtype=np.float64,
    )
    task_distortions = [
        frame_report.task_distortion
        for frame_report in frame_reports
        if frame_report.task_distortion is not None
    ]

    return DeploymentBatchSummary(
        frame_count=len(frame_reports),
        all_roundtrip_equal=all(frame_report.roundtrip_equal for frame_report in frame_reports),
        packet_bits_mean=float(np.mean(packet_bits)),
        packet_bits_std=float(np.std(packet_bits)),
        packet_bits_min=int(np.min(packet_bits)),
        packet_bits_max=int(np.max(packet_bits)),
        rate_proxy_bits_mean=float(np.mean(rate_proxy_bits)),
        rate_proxy_bits_std=float(np.std(rate_proxy_bits)),
        psd_distortion_mean=float(np.mean(psd_distortions)),
        psd_distortion_std=float(np.std(psd_distortions)),
        psd_distortion_min=float(np.min(psd_distortions)),
        psd_distortion_max=float(np.max(psd_distortions)),
        preprocessing_distortion_mean=float(np.mean(preprocessing_distortions)),
        codec_distortion_mean=float(np.mean(codec_distortions)),
        peak_frequency_error_hz_mean=float(np.mean(peak_frequency_errors_hz)),
        peak_frequency_error_hz_max=float(np.max(peak_frequency_errors_hz)),
        peak_power_error_db_mean=float(np.mean(peak_power_errors_db)),
        peak_power_error_db_max=float(np.max(peak_power_errors_db)),
        task_distortion_mean=(
            None if not task_distortions else float(np.mean(np.asarray(task_distortions)))
        ),
    )


def _peak_frequency_error_hz(
    original_frame: FloatArray,
    reconstructed_frame: FloatArray,
    frequency_grid_hz: FloatArray,
) -> float:
    """Return the dominant-peak location error between two aligned PSD frames."""
    original_peak_index = int(np.argmax(original_frame))
    reconstructed_peak_index = int(np.argmax(reconstructed_frame))
    return float(
        abs(
            float(frequency_grid_hz[original_peak_index])
            - float(frequency_grid_hz[reconstructed_peak_index])
        )
    )


def _peak_power_error_db(
    original_frame: FloatArray,
    reconstructed_frame: FloatArray,
) -> float:
    """Return the absolute dominant-peak amplitude error in dB."""
    return float(abs(_to_db(np.max(original_frame)) - _to_db(np.max(reconstructed_frame))))


def _to_db(value: float) -> float:
    """Convert linear power to dB with a small positive floor for stability."""
    return float(10.0 * np.log10(max(float(value), 1.0e-12)))
