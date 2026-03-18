"""Deployment-facing helpers for loading exported codec artifacts and demo samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from codec.config import (
    CodecRuntimeConfig,
    FactorizedEntropyModelConfig,
    PacketFormatConfig,
    PreprocessingConfig,
    ScalarQuantizerConfig,
)
from codec.exceptions import CodecConfigurationError
from data.campaigns import load_campaign_dataset_bundle
from interfaces.api import PsdCodecService
from models.torch_backend import TorchFullCodec
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
        export_dir: Experiment export directory such as `models/exports/baseline_psdcodec`.
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
    dataset_config = artifacts.experiment_config.dataset
    if dataset_config.source_format != "campaigns":
        raise CodecConfigurationError(
            "load_campaign_frame_sample requires campaign-backed deployment assets.",
        )
    if frame_index < 0:
        raise CodecConfigurationError("frame_index must be non-negative.")

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
        max_frames=frame_index + 1,
        noise_floor_window=dataset_config.noise_floor_window,
        noise_floor_percentile=dataset_config.noise_floor_percentile,
    )
    if frame_index >= bundle.frames.shape[0]:
        raise CodecConfigurationError(
            "Requested frame_index "
            f"{frame_index} but only {bundle.frames.shape[0]} frames were loaded."
        )

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
