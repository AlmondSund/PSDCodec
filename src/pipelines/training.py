"""Training, checkpointing, and export orchestration for the PyTorch codec."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import random
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import yaml  # type: ignore[import-untyped]

from codec.config import (
    CodecRuntimeConfig,
    FactorizedEntropyModelConfig,
    PacketFormatConfig,
    PreprocessingConfig,
    ScalarQuantizerConfig,
)
from codec.exceptions import CodecConfigurationError
from codec.preprocessing import FramePreprocessor
from codec.torch_preprocessing import DifferentiableInversePreprocessor
from data.datasets import (
    PreparedPsdBatch,
    PreparedPsdDataset,
    collate_prepared_psd_samples,
)
from models.torch_backend import TorchCodecConfig, TorchFullCodec
from objectives.distortion import IllustrativeTaskConfig, illustrative_task_loss
from objectives.training import (
    RateDistortionLossConfig,
    TrainingLossBreakdown,
    compose_rate_distortion_loss,
    torch_illustrative_task_loss,
)

_torch: Any | None
TorchDataLoader: Any | None
try:
    import torch as _torch
    from torch.utils.data import DataLoader as TorchDataLoader
except ImportError:  # pragma: no cover - exercised only when torch is unavailable
    _torch = None
    TorchDataLoader = None

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

_SELECTION_METRICS: frozenset[str] = frozenset(
    {
        "validation_loss",
        "validation_deployment_score",
        "validation_psd_loss",
        "validation_task_loss",
        "validation_task_monitor",
    }
)

_DEPLOYMENT_SELECTION_COMPONENT_WEIGHTS: dict[str, float] = {
    "psd": 0.35,
    "peak_frequency": 0.45,
    "peak_power": 0.10,
    "task": 0.10,
}

_DEPLOYMENT_SELECTION_COMPONENT_STABILIZERS: dict[str, float] = {
    # These additive floors keep the preprocessing-relative score numerically stable
    # when the deterministic baseline is already very strong on a component.
    "psd": 0.02,
    "peak_frequency": 25_000.0,
    "peak_power": 1.0,
    "task": 0.25,
}

_ACCELERATOR_DEVICE_TYPES: frozenset[str] = frozenset({"cuda", "mps"})


def _require_torch() -> Any:
    """Return the imported torch module or raise a precise error."""
    if _torch is None or TorchDataLoader is None:
        raise ImportError("PyTorch is required to use pipelines.training.")
    return _torch


def _resolve_training_random_seed(
    experiment_config: TrainingExperimentConfig,  # Full experiment configuration
) -> int:
    """Resolve the RNG seed that controls model initialization and batch ordering.

    Purpose:
        Dataset splitting already has its own deterministic seed. Training needs a
        second, explicit seed boundary so reruns with the same experiment config
        produce comparable optimization traces instead of drifting due to model
        initialization or DataLoader shuffle randomness.
    """
    configured_seed = experiment_config.training.random_seed
    if configured_seed is not None:
        return configured_seed
    return experiment_config.dataset.seed


def _seed_training_random_state(
    seed: int,  # Process-wide RNG seed used for this training run
) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducible training starts.

    Side effects:
        Mutates process-wide RNG state before model construction so that parameter
        initialization, stochastic training helpers, and any future Python/NumPy
        randomness start from the same seed on every rerun.

    Trade-off:
        This helper intentionally does not enable deterministic kernels globally.
        Full deterministic algorithms would reduce accelerator throughput, while
        simple seed control is sufficient to make experiment comparisons fair.
    """
    torch_module = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if hasattr(torch_module, "cuda"):
        torch_module.cuda.manual_seed_all(seed)


def _seed_data_loader_worker(
    worker_index: int,  # Worker id assigned by the DataLoader runtime
) -> None:
    """Seed per-worker Python and NumPy RNGs from PyTorch's worker-local seed."""
    del worker_index  # The worker-local torch seed already encodes the worker identity.
    torch_module = _require_torch()
    worker_seed = int(torch_module.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _raise_if_non_finite_tensors(
    named_values: dict[str, Tensor],  # Named tensors whose entries must all be finite
) -> None:
    """Raise a precise floating-point error when any tensor contains NaN or Inf.

    Purpose:
        Training still needs explicit non-finite guards, but calling `.item()` for
        every intermediate tensor forces repeated accelerator synchronizations. This
        helper reduces the common-case overhead to one device-to-host check, then
        falls back to per-tensor diagnosis only on the rare failure path.
    """
    torch_module = _require_torch()
    if not named_values:
        return

    finite_checks = torch_module.stack(
        [torch_module.isfinite(value).all() for value in named_values.values()]
    )
    if bool(finite_checks.all().item()):
        return

    non_finite_names = [
        name
        for name, value in named_values.items()
        if not bool(torch_module.isfinite(value).all().item())
    ]
    joined_names = ", ".join(non_finite_names)
    raise FloatingPointError(
        "The training path produced non-finite tensors before exact task monitoring. "
        f"Affected tensors: {joined_names}.",
    )


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset loading configuration for one training experiment."""

    dataset_path: Path  # Input `.npz` archive or raw campaign root directory
    source_format: str = "npz"  # Dataset source type: `npz` or `campaigns`
    frames_key: str = "frames"  # NPZ key containing the PSD frame matrix
    frequency_grid_key: str | None = "frequency_grid_hz"  # Optional NPZ key for the frequency grid
    noise_floor_key: str | None = None  # Optional NPZ key for explicit noise floors
    noise_floor_window: int | None = None  # Optional history window used to estimate noise floors
    noise_floor_percentile: float = 10.0  # Robust percentile used for history-based noise floors
    validation_fraction: float = 0.2  # Fraction of samples assigned to validation
    shuffle: bool = (
        True  # Whether to shuffle before splitting and when building the training loader
    )
    seed: int = 0  # Random seed for deterministic train/validation splits
    campaign_include_globs: list[str] = field(default_factory=lambda: ["*"])
    campaign_exclude_globs: list[str] = field(default_factory=list)
    campaign_node_globs: list[str] = field(default_factory=lambda: ["Node*.csv"])
    campaign_target_bin_count: int | None = None  # Optional common PSD length after harmonization
    campaign_value_scale: str = "db_to_power"  # Raw-value transform applied at campaign ingestion
    campaign_max_frames: int | None = None  # Optional deterministic truncation after raw loading

    def __post_init__(self) -> None:
        """Validate dataset-loading configuration."""
        if self.source_format not in {"npz", "campaigns"}:
            raise CodecConfigurationError("source_format must be either 'npz' or 'campaigns'.")
        if not (0.0 < self.validation_fraction < 1.0):
            raise CodecConfigurationError(
                "validation_fraction must lie in the open interval (0, 1)."
            )
        if self.noise_floor_window is not None and self.noise_floor_window <= 0:
            raise CodecConfigurationError("noise_floor_window must be strictly positive.")
        if not (0.0 <= self.noise_floor_percentile <= 100.0):
            raise CodecConfigurationError("noise_floor_percentile must lie in [0, 100].")
        if self.campaign_target_bin_count is not None and self.campaign_target_bin_count <= 0:
            raise CodecConfigurationError(
                "campaign_target_bin_count must be strictly positive when set."
            )
        if self.campaign_max_frames is not None and self.campaign_max_frames <= 0:
            raise CodecConfigurationError("campaign_max_frames must be strictly positive when set.")
        if self.campaign_value_scale not in {"db_to_power", "identity"}:
            raise CodecConfigurationError(
                "campaign_value_scale must be either 'db_to_power' or 'identity'."
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and runtime settings for one training run."""

    epoch_count: int = 10  # Number of passes over the training set
    batch_size: int = 32  # Mini-batch size
    learning_rate: float = 1.0e-3  # Adam learning rate
    weight_decay: float = 0.0  # Adam weight decay
    gradient_clip_norm: float | None = 1.0  # Optional global gradient clipping threshold
    device: str = "auto"  # Torch device string or `auto`
    mixed_precision: str = "auto"  # `auto`, `disabled`, `fp16`, or `bf16`
    enable_model_compile: bool = False  # Whether to `torch.compile` the training graph
    allow_tf32: bool = True  # Enable TensorFloat-32 matmul kernels on supported CUDA GPUs
    data_loader_worker_count: int | None = None  # `None` means resolve a worker count automatically
    pin_memory: bool | None = None  # `None` means pin host batches only for CUDA runs
    prefetch_factor: int | None = None  # Optional DataLoader prefetch depth when workers > 0
    persistent_data_loader_workers: bool = (
        True  # Keep DataLoader workers alive across epochs when workers > 0
    )
    random_seed: int | None = None  # Optional RNG seed for model init and mini-batch order
    loss: RateDistortionLossConfig = RateDistortionLossConfig()  # Lagrangian loss weights

    def __post_init__(self) -> None:
        """Validate optimizer hyperparameters."""
        if self.epoch_count <= 0:
            raise CodecConfigurationError("epoch_count must be strictly positive.")
        if self.batch_size <= 0:
            raise CodecConfigurationError("batch_size must be strictly positive.")
        if self.learning_rate <= 0.0:
            raise CodecConfigurationError("learning_rate must be strictly positive.")
        if self.weight_decay < 0.0:
            raise CodecConfigurationError("weight_decay must be non-negative.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise CodecConfigurationError("gradient_clip_norm must be strictly positive when set.")
        if not self.device:
            raise CodecConfigurationError("device must be a non-empty string.")
        if self.mixed_precision not in {"auto", "disabled", "fp16", "bf16"}:
            raise CodecConfigurationError(
                "mixed_precision must be one of {'auto', 'disabled', 'fp16', 'bf16'}."
            )
        if self.data_loader_worker_count is not None and self.data_loader_worker_count < 0:
            raise CodecConfigurationError(
                "data_loader_worker_count must be non-negative when set."
            )
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise CodecConfigurationError("prefetch_factor must be strictly positive when set.")
        if self.random_seed is not None and self.random_seed < 0:
            raise CodecConfigurationError("random_seed must be non-negative when set.")


@dataclass(frozen=True)
class ArtifactConfig:
    """Output locations and export switches for one experiment."""

    experiment_name: str  # Stable artifact namespace under models/checkpoints and models/exports
    checkpoint_root: Path = Path("models/checkpoints")  # Root directory for checkpoints
    export_root: Path = Path("models/exports")  # Root directory for export-ready artifacts
    export_onnx: bool = True  # Whether to export the encoder boundary to ONNX after training
    save_latest_checkpoint: bool = True  # Whether to persist the latest checkpoint every epoch
    latest_checkpoint_interval: int = 1  # Save the latest checkpoint every N epochs
    save_best_checkpoint: bool = True  # Whether to persist the best validation checkpoint
    selection_metric: str = "validation_loss"  # Metric used to pick the best checkpoint
    require_selection_to_beat_preprocessing: bool = (
        False  # Reject best-checkpoint candidates that do not beat preprocessing-only
    )

    def __post_init__(self) -> None:
        """Validate artifact and checkpoint-selection settings."""
        if self.selection_metric not in _SELECTION_METRICS:
            raise CodecConfigurationError(
                "selection_metric must be one of "
                f"{sorted(_SELECTION_METRICS)}.",
            )
        if self.latest_checkpoint_interval <= 0:
            raise CodecConfigurationError("latest_checkpoint_interval must be strictly positive.")


@dataclass(frozen=True)
class TrainingExperimentConfig:
    """Full experiment configuration for dataset loading, training, and export."""

    dataset: DatasetConfig
    runtime: CodecRuntimeConfig
    model: TorchCodecConfig
    training: TrainingConfig
    artifacts: ArtifactConfig
    task: IllustrativeTaskConfig | None = None

    def __post_init__(self) -> None:
        """Validate cross-component architectural consistency."""
        if self.runtime.entropy_model.alphabet_size != self.model.codebook_size:
            raise CodecConfigurationError(
                "runtime entropy alphabet_size must match model codebook_size.",
            )

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,  # YAML file describing the full experiment
    ) -> TrainingExperimentConfig:
        """Load an experiment configuration from a YAML file."""
        with Path(config_path).open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        if not isinstance(payload, dict):
            raise CodecConfigurationError(
                "Training experiment YAML must contain a mapping at the root."
            )
        return cls.from_dict(payload)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],  # Nested experiment configuration dictionary
    ) -> TrainingExperimentConfig:
        """Build an experiment configuration from a nested dictionary."""
        dataset_payload = _expect_mapping(payload, "dataset")
        runtime_payload = _expect_mapping(payload, "runtime")
        model_payload = _expect_mapping(payload, "model")
        training_payload = _expect_mapping(payload, "training")
        artifacts_payload = _expect_mapping(payload, "artifacts")
        task_payload = payload.get("task")
        task_config = (
            None
            if task_payload is None
            else IllustrativeTaskConfig(**_coerce_mapping(task_payload))
        )
        return cls(
            dataset=DatasetConfig(**_coerce_path_fields(dataset_payload, {"dataset_path"})),
            runtime=_parse_runtime_config(runtime_payload),
            model=TorchCodecConfig(**_coerce_mapping(model_payload)),
            training=_parse_training_config(training_payload),
            artifacts=ArtifactConfig(
                **_coerce_path_fields(
                    artifacts_payload,
                    {"checkpoint_root", "export_root"},
                ),
            ),
            task=task_config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the experiment configuration into a JSON/YAML-safe mapping."""
        return {
            "dataset": _dataclass_to_dict(self.dataset),
            "runtime": _runtime_config_to_dict(self.runtime),
            "model": _dataclass_to_dict(self.model),
            "training": {
                **_dataclass_to_dict(self.training),
                "loss": _dataclass_to_dict(self.training.loss),
            },
            "artifacts": _dataclass_to_dict(self.artifacts),
            "task": None if self.task is None else _dataclass_to_dict(self.task),
        }


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated training and validation metrics for one epoch."""

    epoch_index: int
    training_loss: float
    validation_loss: float
    training_psd_loss: float
    validation_psd_loss: float
    training_rate_bits: float
    validation_rate_bits: float
    training_vq_loss: float
    validation_vq_loss: float
    training_task_loss: float
    validation_task_loss: float
    validation_task_monitor: float | None = None
    validation_preprocessing_psd_loss: float | None = None
    validation_preprocessing_task_monitor: float | None = None
    validation_peak_frequency_error_hz: float | None = None
    validation_peak_power_error_db: float | None = None
    validation_preprocessing_peak_frequency_error_hz: float | None = None
    validation_preprocessing_peak_power_error_db: float | None = None
    validation_deployment_score: float | None = None


@dataclass(frozen=True)
class TrainingSummary:
    """Training outputs and persisted artifact locations."""

    history: tuple[EpochMetrics, ...]
    best_epoch_index: int
    selection_metric: str
    best_selection_score: float
    best_validation_loss: float
    resolved_training_device: str
    best_checkpoint_path: Path | None
    latest_checkpoint_path: Path | None
    export_dir: Path
    runtime_asset_dir: Path
    onnx_path: Path | None


@dataclass(frozen=True)
class EpochProgressUpdate:
    """Progress snapshot emitted after one completed training epoch."""

    epoch_metrics: EpochMetrics  # Aggregated metrics for the completed epoch
    completed_epoch_count: int  # Number of epochs completed so far
    total_epoch_count: int  # Total epochs requested by the experiment
    remaining_epoch_count: int  # Epochs still pending after this update
    selection_metric: str  # Metric used to choose the best checkpoint
    current_selection_score: float | None  # Current epoch score for the selected metric
    selection_candidate_accepted: bool  # Whether this epoch may replace the persisted best
    best_selection_score: float  # Best score seen so far for the selected metric
    best_validation_loss: float  # Best validation loss observed so far


@dataclass(frozen=True)
class LoadedTrainingCheckpoint:
    """Typed view over a persisted training checkpoint."""

    epoch_index: int
    selection_metric: str
    best_selection_score: float
    best_validation_loss: float
    experiment_config: TrainingExperimentConfig
    metrics: EpochMetrics
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]


def _write_runtime_assets_from_model(
    model: TorchFullCodec,  # Trained codec whose runtime tensors should be exported
    runtime_asset_dir: Path,  # Destination `runtime_assets/` directory
    runtime_config: CodecRuntimeConfig,  # Deterministic runtime configuration JSON payload
) -> None:
    """Persist the deterministic runtime bundle for deployment and notebook use.

    Purpose:
        The deployment service needs the learned codebook, entropy probabilities, and
        preprocessing/runtime configuration independently of the training loop. This
        helper keeps that boundary reusable for both immediate post-training export
        and later checkpoint recovery.
    """
    runtime_asset_dir.mkdir(parents=True, exist_ok=True)
    np.save(runtime_asset_dir / "codebook.npy", model.export_runtime_codebook())
    np.save(
        runtime_asset_dir / "entropy_probabilities.npy",
        model.export_runtime_probabilities(),
    )
    (runtime_asset_dir / "runtime_config.json").write_text(
        json.dumps(_runtime_config_to_dict(runtime_config), indent=2),
        encoding="utf-8",
    )


def _write_training_summary_metadata(
    metadata_path: Path,  # Destination `training_summary.json`
    *,
    experiment_config: TrainingExperimentConfig,  # Full experiment configuration
    resolved_training_device: str,  # Device string recorded for this export
    selection_metric: str,  # Metric used to select the best checkpoint
    best_epoch_index: int,  # Zero-based epoch index of the best checkpoint
    best_selection_score: float,  # Best score observed under `selection_metric`
    best_validation_loss: float,  # Validation loss at the selected checkpoint
    history: Sequence[EpochMetrics],  # Epoch history included in the summary JSON
    best_checkpoint_path: Path | None,  # Persisted best-checkpoint location
    latest_checkpoint_path: Path | None,  # Optional latest-checkpoint location
    runtime_asset_dir: Path,  # Exported runtime-asset directory
    onnx_path: Path | None,  # Optional ONNX encoder path
) -> None:
    """Write the deployment summary JSON consumed by notebook/runtime loaders.

    Purpose:
        The deployment bundle needs one stable metadata file that ties together the
        exported runtime tensors, the ONNX encoder boundary, and the checkpoint path
        used to restore the server-side decoder. Centralizing the write logic keeps
        fresh training exports and recovered exports structurally identical.
    """
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "experiment_config": experiment_config.to_dict(),
                "resolved_training_device": resolved_training_device,
                "selection_metric": selection_metric,
                "best_epoch_index": best_epoch_index,
                "best_selection_score": best_selection_score,
                "best_validation_loss": best_validation_loss,
                "history": [asdict(epoch) for epoch in history],
                "best_checkpoint_path": None
                if best_checkpoint_path is None
                else str(best_checkpoint_path),
                "latest_checkpoint_path": None
                if latest_checkpoint_path is None
                else str(latest_checkpoint_path),
                "runtime_asset_dir": str(runtime_asset_dir),
                "onnx_path": None if onnx_path is None else str(onnx_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _resolve_recovery_source_config_path(
    checkpoint_path: Path,  # Checkpoint used for export recovery
    *,
    experiment_name: str,  # Canonical experiment name used for YAML copies
    source_config_path: Path | None,  # Explicit YAML override, if any
) -> Path | None:
    """Resolve which YAML file should be copied into a recovered export directory."""
    if source_config_path is not None:
        return source_config_path
    inferred_path = checkpoint_path.parent / f"{experiment_name}.yaml"
    if inferred_path.exists():
        return inferred_path
    return None


def _write_resolved_experiment_config_yaml(
    export_dir: Path,  # Export directory that should contain the canonical resolved YAML
    experiment_config: TrainingExperimentConfig,  # Exact configuration used for training
) -> Path:
    """Write the resolved experiment YAML consumed by stale-artifact checks.

    Purpose:
        The deployment loader compares `training_summary.json` against the YAML copy
        stored beside the export bundle. That YAML therefore must reflect the
        resolved configuration used during training, not the pre-adjustment source
        file that may still reference raw campaigns or `device: auto`.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    destination = export_dir / f"{experiment_config.artifacts.experiment_name}.yaml"
    destination.write_text(
        yaml.safe_dump(experiment_config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    return destination


def _copy_source_config_sidecar_if_present(
    destination_dir: Path,  # Directory that should receive the optional sidecar copy
    *,
    experiment_name: str,  # Experiment name used to build a stable sidecar filename
    source_config_path: Path | None,  # Human-authored source YAML, if available
) -> Path | None:
    """Copy the original source YAML as a non-canonical sidecar when available.

    Purpose:
        The raw source YAML is still useful for debugging, but it must not replace
        the canonical resolved export YAML because the deployment loader uses the
        latter for stale-artifact detection. This helper keeps the human-authored
        YAML as an auxiliary sidecar instead.
    """
    if source_config_path is None:
        return None
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = (
        destination_dir
        / f"{experiment_name}.source{source_config_path.suffix or '.yaml'}"
    )
    shutil.copy2(source_config_path, destination)
    return destination


def recover_training_export_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    export_dir: str | Path | None = None,
    source_config_path: str | Path | None = None,
    resolved_training_device: str = "recovered_from_checkpoint",
) -> TrainingSummary:
    """Rebuild deployment artifacts from a saved checkpoint without retraining.

    Purpose:
        Training can finish successfully and still fail later while exporting ONNX or
        writing the deployment bundle. The saved checkpoint remains the authoritative
        model state, so this helper reconstructs the export directory directly from
        that checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint used as the recovery source.
        export_dir: Optional explicit export directory override. When omitted, the
            helper uses `artifacts.export_root / artifacts.experiment_name` from the
            checkpointed experiment configuration.
        source_config_path: Optional YAML file copied into the recovered export
            directory for stale-artifact detection in notebook/deployment loaders.
        resolved_training_device: Human-readable device label written into the
            recovered `training_summary.json`.

    Returns:
        A `TrainingSummary` describing the recovered export bundle.

    Raises:
        FileNotFoundError: If `checkpoint_path` does not exist.
    """
    resolved_checkpoint_path = Path(checkpoint_path)
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"Training checkpoint does not exist: {resolved_checkpoint_path}")

    loaded_checkpoint = load_training_checkpoint(
        resolved_checkpoint_path,
        map_location="cpu",
    )
    resolved_export_dir = (
        Path(export_dir)
        if export_dir is not None
        else (
            loaded_checkpoint.experiment_config.artifacts.export_root
            / loaded_checkpoint.experiment_config.artifacts.experiment_name
        )
    )
    runtime_asset_dir = resolved_export_dir / "runtime_assets"
    resolved_export_dir.mkdir(parents=True, exist_ok=True)

    source_config = _resolve_recovery_source_config_path(
        resolved_checkpoint_path,
        experiment_name=loaded_checkpoint.experiment_config.artifacts.experiment_name,
        source_config_path=None if source_config_path is None else Path(source_config_path),
    )
    _write_resolved_experiment_config_yaml(
        resolved_export_dir,
        loaded_checkpoint.experiment_config,
    )
    _copy_source_config_sidecar_if_present(
        resolved_export_dir,
        experiment_name=loaded_checkpoint.experiment_config.artifacts.experiment_name,
        source_config_path=source_config,
    )

    model = TorchFullCodec(loaded_checkpoint.experiment_config.model)
    model.load_state_dict(loaded_checkpoint.model_state_dict)
    model.eval()

    _write_runtime_assets_from_model(
        model,
        runtime_asset_dir,
        loaded_checkpoint.experiment_config.runtime,
    )
    recovered_onnx_path = None
    if loaded_checkpoint.experiment_config.artifacts.export_onnx:
        recovered_onnx_path = resolved_export_dir / "encoder.onnx"
        model.export_encoder_to_onnx(recovered_onnx_path)

    inferred_latest_checkpoint_path = resolved_checkpoint_path.parent / "latest.pt"
    latest_checkpoint_path = (
        inferred_latest_checkpoint_path if inferred_latest_checkpoint_path.exists() else None
    )
    history = (loaded_checkpoint.metrics,)
    _write_training_summary_metadata(
        resolved_export_dir / "training_summary.json",
        experiment_config=loaded_checkpoint.experiment_config,
        resolved_training_device=resolved_training_device,
        selection_metric=loaded_checkpoint.selection_metric,
        best_epoch_index=loaded_checkpoint.epoch_index,
        best_selection_score=loaded_checkpoint.best_selection_score,
        best_validation_loss=loaded_checkpoint.best_validation_loss,
        history=history,
        best_checkpoint_path=resolved_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        runtime_asset_dir=runtime_asset_dir,
        onnx_path=recovered_onnx_path,
    )
    return TrainingSummary(
        history=history,
        best_epoch_index=loaded_checkpoint.epoch_index,
        selection_metric=loaded_checkpoint.selection_metric,
        best_selection_score=loaded_checkpoint.best_selection_score,
        best_validation_loss=loaded_checkpoint.best_validation_loss,
        resolved_training_device=resolved_training_device,
        best_checkpoint_path=resolved_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        export_dir=resolved_export_dir,
        runtime_asset_dir=runtime_asset_dir,
        onnx_path=recovered_onnx_path,
    )


class TorchCodecTrainer:
    """Application-layer trainer for the PyTorch PSD codec."""

    def __init__(self, experiment_config: TrainingExperimentConfig) -> None:
        """Initialize model, optimizer, and deterministic preprocessing helpers."""
        torch_module = _require_torch()
        self.experiment_config = experiment_config
        self.preprocessor = FramePreprocessor(experiment_config.runtime.preprocessing)
        self._training_random_seed = _resolve_training_random_seed(experiment_config)
        _seed_training_random_state(self._training_random_seed)
        self.training_device = _resolve_training_device_string(experiment_config.training.device)
        self._training_device_handle = torch_module.device(self.training_device)
        self._training_device_type = self._training_device_handle.type
        self._validation_diagnostics_required = _requires_exact_validation_diagnostics(
            experiment_config
        )
        self._pin_memory = _resolve_pin_memory_enabled(
            experiment_config.training.pin_memory,
            device_type=self._training_device_type,
        )
        self._data_loader_worker_count = _resolve_data_loader_worker_count(
            experiment_config.training.data_loader_worker_count,
            device_type=self._training_device_type,
        )
        self._autocast_dtype = _resolve_autocast_dtype(
            experiment_config.training.mixed_precision,
            device=self._training_device_handle,
        )
        self._autocast_enabled = self._autocast_dtype is not None
        self.model = TorchFullCodec(experiment_config.model).to(self.training_device)
        self.training_model = self.model
        if (
            experiment_config.training.enable_model_compile
            and self._training_device_type in _ACCELERATOR_DEVICE_TYPES
            and hasattr(torch_module, "compile")
        ):
            self.training_model = torch_module.compile(
                self.model,
                mode="reduce-overhead",
            )
        self.optimizer = torch_module.optim.Adam(
            self.model.parameters(),
            lr=experiment_config.training.learning_rate,
            weight_decay=experiment_config.training.weight_decay,
        )
        grad_scaler_cls = getattr(torch_module.amp, "GradScaler", None)
        self._grad_scaler = (
            None
            if grad_scaler_cls is None or self._autocast_dtype != torch_module.float16
            else grad_scaler_cls("cuda", enabled=self._training_device_type == "cuda")
        )
        if self._training_device_type == "cuda":
            torch_module.backends.cuda.matmul.allow_tf32 = experiment_config.training.allow_tf32
            if hasattr(torch_module.backends, "cudnn"):
                torch_module.backends.cudnn.allow_tf32 = experiment_config.training.allow_tf32
            if hasattr(torch_module, "set_float32_matmul_precision"):
                torch_module.set_float32_matmul_precision(
                    "high" if experiment_config.training.allow_tf32 else "highest"
                )

    def load_prepared_datasets(self) -> tuple[PreparedPsdDataset, PreparedPsdDataset]:
        """Load the configured dataset, preprocess it, and split it into train/validation sets."""
        dataset_config = self.experiment_config.dataset
        if dataset_config.source_format == "npz":
            dataset = PreparedPsdDataset.from_npz(
                dataset_config.dataset_path,
                preprocessor=self.preprocessor,
                frames_key=dataset_config.frames_key,
                frequency_grid_key=dataset_config.frequency_grid_key,
                noise_floor_key=dataset_config.noise_floor_key,
                noise_floor_window=dataset_config.noise_floor_window,
                noise_floor_percentile=dataset_config.noise_floor_percentile,
            )
        else:
            dataset = PreparedPsdDataset.from_campaigns(
                dataset_config.dataset_path,
                preprocessor=self.preprocessor,
                include_campaign_globs=dataset_config.campaign_include_globs,
                exclude_campaign_globs=dataset_config.campaign_exclude_globs,
                include_node_globs=dataset_config.campaign_node_globs,
                target_bin_count=dataset_config.campaign_target_bin_count,
                value_scale=dataset_config.campaign_value_scale,
                max_frames=dataset_config.campaign_max_frames,
                noise_floor_window=dataset_config.noise_floor_window,
                noise_floor_percentile=dataset_config.noise_floor_percentile,
            )
        if dataset.reduced_bin_count != self.experiment_config.model.reduced_bin_count:
            raise CodecConfigurationError(
                "Prepared dataset reduced_bin_count does not match the Torch model input size.",
            )
        if self.experiment_config.task is not None:
            if dataset.frequency_grid_hz is None:
                raise CodecConfigurationError(
                    "Illustrative task training requires dataset.frequency_grid_hz."
                )
            if dataset.noise_floors is None:
                raise CodecConfigurationError(
                    "Illustrative task training requires dataset.noise_floors."
                )
        return dataset.train_validation_split(
            validation_fraction=dataset_config.validation_fraction,
            seed=dataset_config.seed,
            shuffle=dataset_config.shuffle,
        )

    def fit(
        self,
        training_dataset: PreparedPsdDataset,
        validation_dataset: PreparedPsdDataset,
        *,
        source_config_path: Path | None = None,
        progress_reporter: Callable[[EpochProgressUpdate], None] | None = None,
    ) -> TrainingSummary:
        """Train the codec, persist checkpoints, and export runtime artifacts.

        Args:
            training_dataset: Prepared dataset used for optimization.
            validation_dataset: Prepared dataset used for model selection.
            source_config_path: Optional YAML path copied into the artifact directories.
            progress_reporter: Optional callback invoked after each completed epoch.
                This keeps terminal/UI reporting outside the trainer core while still
                exposing epoch-by-epoch progress to CLI entrypoints.
        """
        if TorchDataLoader is None:
            raise ImportError("PyTorch is required to construct training data loaders.")
        torch_module = _require_torch()
        if training_dataset.original_bin_count != validation_dataset.original_bin_count:
            raise CodecConfigurationError(
                "Training and validation datasets must share the same original frame length."
            )
        inverse_preprocessor = DifferentiableInversePreprocessor(
            self.experiment_config.runtime.preprocessing,
            training_dataset.original_bin_count,
        )
        if inverse_preprocessor.reduced_bin_count != self.experiment_config.model.reduced_bin_count:
            raise CodecConfigurationError(
                "Differentiable inverse preprocessing width does not match the model width.",
            )

        data_loader_kwargs: dict[str, Any] = {
            "batch_size": self.experiment_config.training.batch_size,
            "collate_fn": collate_prepared_psd_samples,
            "num_workers": self._data_loader_worker_count,
        }
        if self._data_loader_worker_count > 0:
            data_loader_kwargs["persistent_workers"] = (
                self.experiment_config.training.persistent_data_loader_workers
            )
            resolved_prefetch_factor = (
                2
                if self.experiment_config.training.prefetch_factor is None
                else self.experiment_config.training.prefetch_factor
            )
            data_loader_kwargs["prefetch_factor"] = resolved_prefetch_factor
            data_loader_kwargs["worker_init_fn"] = _seed_data_loader_worker

        train_loader_generator = torch_module.Generator()
        train_loader_generator.manual_seed(self._training_random_seed)

        train_loader: Any = TorchDataLoader(
            training_dataset,
            shuffle=self.experiment_config.dataset.shuffle,
            generator=train_loader_generator,
            **data_loader_kwargs,
        )
        validation_loader: Any = TorchDataLoader(
            validation_dataset,
            shuffle=False,
            **data_loader_kwargs,
        )
        side_information_bits = float(
            self.experiment_config.runtime.preprocessing.block_count
            * self.experiment_config.runtime.preprocessing.side_information_bits_per_block
        )
        validation_baseline_metrics = (
            None
            if not self._validation_diagnostics_required
            else self._compute_validation_baseline_metrics(
                validation_dataset,
                inverse_preprocessor=inverse_preprocessor,
            )
        )

        checkpoint_dir = (
            self.experiment_config.artifacts.checkpoint_root
            / self.experiment_config.artifacts.experiment_name
        )
        export_dir = (
            self.experiment_config.artifacts.export_root
            / self.experiment_config.artifacts.experiment_name
        )
        runtime_asset_dir = export_dir / "runtime_assets"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        runtime_asset_dir.mkdir(parents=True, exist_ok=True)
        if source_config_path is not None:
            shutil.copy2(source_config_path, checkpoint_dir / source_config_path.name)
        _write_resolved_experiment_config_yaml(export_dir, self.experiment_config)
        _copy_source_config_sidecar_if_present(
            export_dir,
            experiment_name=self.experiment_config.artifacts.experiment_name,
            source_config_path=source_config_path,
        )

        history: list[EpochMetrics] = []
        best_validation_loss = float("inf")
        best_selection_score = float("inf")
        best_epoch_index = -1
        best_checkpoint_path: Path | None = None
        latest_checkpoint_path: Path | None = None
        best_model_state_dict: dict[str, Any] | None = None

        for epoch_index in range(self.experiment_config.training.epoch_count):
            training_metrics = self._run_epoch(
                train_loader,
                inverse_preprocessor=inverse_preprocessor,
                side_information_bits=side_information_bits,
                training=True,
                dataset_frequency_grid_hz=training_dataset.frequency_grid_hz,
            )
            validation_metrics = self._run_epoch(
                validation_loader,
                inverse_preprocessor=inverse_preprocessor,
                side_information_bits=side_information_bits,
                training=False,
                dataset_frequency_grid_hz=validation_dataset.frequency_grid_hz,
            )
            validation_deployment_score = _compose_validation_deployment_score(
                validation_psd_loss=validation_metrics.psd_loss,
                validation_preprocessing_psd_loss=(
                    None
                    if validation_baseline_metrics is None
                    else validation_baseline_metrics.preprocessing_psd_loss
                ),
                validation_peak_frequency_error_hz=validation_metrics.peak_frequency_error_hz,
                validation_preprocessing_peak_frequency_error_hz=(
                    None
                    if validation_baseline_metrics is None
                    else validation_baseline_metrics.preprocessing_peak_frequency_error_hz
                ),
                validation_peak_power_error_db=validation_metrics.peak_power_error_db,
                validation_preprocessing_peak_power_error_db=(
                    None
                    if validation_baseline_metrics is None
                    else validation_baseline_metrics.preprocessing_peak_power_error_db
                ),
                validation_task_monitor=validation_metrics.task_monitor,
                validation_preprocessing_task_monitor=(
                    None
                    if validation_baseline_metrics is None
                    else validation_baseline_metrics.preprocessing_task_monitor
                ),
            )
            epoch_metrics = EpochMetrics(
                epoch_index=epoch_index,
                training_loss=training_metrics.total_loss,
                validation_loss=validation_metrics.total_loss,
                training_psd_loss=training_metrics.psd_loss,
                validation_psd_loss=validation_metrics.psd_loss,
                training_rate_bits=training_metrics.rate_bits
                + training_metrics.side_information_bits,
                validation_rate_bits=validation_metrics.rate_bits
                + validation_metrics.side_information_bits,
                training_vq_loss=training_metrics.vq_loss,
                validation_vq_loss=validation_metrics.vq_loss,
                training_task_loss=training_metrics.task_loss,
                validation_task_loss=validation_metrics.task_loss,
                validation_task_monitor=validation_metrics.task_monitor,
                validation_preprocessing_psd_loss=(
                    None
                    if validation_baseline_metrics is None
                    else validation_baseline_metrics.preprocessing_psd_loss
                ),
                validation_preprocessing_task_monitor=(
                    None
                    if validation_baseline_metrics is None
                    else validation_baseline_metrics.preprocessing_task_monitor
                ),
                validation_peak_frequency_error_hz=validation_metrics.peak_frequency_error_hz,
                validation_peak_power_error_db=validation_metrics.peak_power_error_db,
                validation_preprocessing_peak_frequency_error_hz=(
                    None
                    if validation_baseline_metrics is None
                    else validation_baseline_metrics.preprocessing_peak_frequency_error_hz
                ),
                validation_preprocessing_peak_power_error_db=(
                    None
                    if validation_baseline_metrics is None
                    else validation_baseline_metrics.preprocessing_peak_power_error_db
                ),
                validation_deployment_score=validation_deployment_score,
            )
            history.append(epoch_metrics)
            selection_score = _resolve_epoch_selection_score(
                epoch_metrics,
                selection_metric=self.experiment_config.artifacts.selection_metric,
            )
            is_acceptable_candidate = _selection_candidate_is_acceptable(
                epoch_metrics,
                require_selection_to_beat_preprocessing=(
                    self.experiment_config.artifacts.require_selection_to_beat_preprocessing
                ),
            )
            if selection_score < best_selection_score and is_acceptable_candidate:
                best_selection_score = selection_score
                best_epoch_index = epoch_index
                best_validation_loss = epoch_metrics.validation_loss
                best_model_state_dict = copy.deepcopy(self.model.state_dict())
                if self.experiment_config.artifacts.save_best_checkpoint:
                    best_checkpoint_path = checkpoint_dir / "best.pt"
                    self._save_checkpoint(
                        best_checkpoint_path,
                        epoch_metrics=epoch_metrics,
                        best_selection_score=best_selection_score,
                        best_validation_loss=best_validation_loss,
                    )

            latest_checkpoint_path = checkpoint_dir / "latest.pt"
            should_save_latest = (
                self.experiment_config.artifacts.save_latest_checkpoint
                and (
                    (epoch_index + 1)
                    % self.experiment_config.artifacts.latest_checkpoint_interval
                    == 0
                    or epoch_index + 1 == self.experiment_config.training.epoch_count
                )
            )
            if should_save_latest:
                self._save_checkpoint(
                    latest_checkpoint_path,
                    epoch_metrics=epoch_metrics,
                    best_selection_score=best_selection_score,
                    best_validation_loss=best_validation_loss,
                )
            if progress_reporter is not None:
                progress_reporter(
                    EpochProgressUpdate(
                        epoch_metrics=epoch_metrics,
                        completed_epoch_count=epoch_index + 1,
                        total_epoch_count=self.experiment_config.training.epoch_count,
                        remaining_epoch_count=(
                            self.experiment_config.training.epoch_count - epoch_index - 1
                        ),
                        selection_metric=self.experiment_config.artifacts.selection_metric,
                        current_selection_score=selection_score,
                        selection_candidate_accepted=is_acceptable_candidate,
                        best_selection_score=best_selection_score,
                        best_validation_loss=best_validation_loss,
                    )
                )

        if best_model_state_dict is None:
            if self.experiment_config.artifacts.require_selection_to_beat_preprocessing:
                raise CodecConfigurationError(
                    "No validation epoch beat the preprocessing-only baseline under "
                    "the configured deployment-aligned selection guard."
                )
            raise CodecConfigurationError(
                "Training completed without selecting any checkpoint as the best model."
            )
        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
        self._export_runtime_assets(runtime_asset_dir)
        onnx_path = None
        if self.experiment_config.artifacts.export_onnx:
            onnx_path = export_dir / "encoder.onnx"
            self.model.export_encoder_to_onnx(onnx_path)
        _write_training_summary_metadata(
            export_dir / "training_summary.json",
            experiment_config=self.experiment_config,
            resolved_training_device=self.training_device,
            selection_metric=self.experiment_config.artifacts.selection_metric,
            best_epoch_index=best_epoch_index,
            best_selection_score=best_selection_score,
            best_validation_loss=best_validation_loss,
            history=history,
            best_checkpoint_path=best_checkpoint_path,
            latest_checkpoint_path=latest_checkpoint_path,
            runtime_asset_dir=runtime_asset_dir,
            onnx_path=onnx_path,
        )
        return TrainingSummary(
            history=tuple(history),
            best_epoch_index=best_epoch_index,
            selection_metric=self.experiment_config.artifacts.selection_metric,
            best_selection_score=best_selection_score,
            best_validation_loss=best_validation_loss,
            resolved_training_device=self.training_device,
            best_checkpoint_path=best_checkpoint_path,
            latest_checkpoint_path=latest_checkpoint_path,
            export_dir=export_dir,
            runtime_asset_dir=runtime_asset_dir,
            onnx_path=onnx_path,
        )

    def _run_epoch(
        self,
        loader: Any,
        *,
        inverse_preprocessor: DifferentiableInversePreprocessor,
        side_information_bits: float,
        training: bool,
        dataset_frequency_grid_hz: np.ndarray | None,
    ) -> _FinalizedMetrics:
        """Run one train or validation epoch and aggregate scalar metrics."""
        torch_module = _require_torch()
        self.model.train(mode=training)
        aggregated = _AggregatedMetrics()
        task_frequency_grid_hz = None
        if self.experiment_config.task is not None:
            if dataset_frequency_grid_hz is None:
                raise CodecConfigurationError(
                    "Illustrative task training requires dataset_frequency_grid_hz."
                )
            task_frequency_grid_hz = torch_module.as_tensor(
                dataset_frequency_grid_hz,
                dtype=torch_module.float32,
                device=self.training_device,
            )

        for batch in loader:
            tensor_batch = self._batch_to_tensors(batch)
            with torch_module.set_grad_enabled(training):
                with self._autocast_context():
                    output = self.training_model(tensor_batch.normalized_frames)
                reconstructed_normalized_frames = output.reconstructed_normalized_frames.to(
                    dtype=torch_module.float32
                )
                rate_bits = output.rate_bits.to(dtype=torch_module.float32)
                vq_loss = output.vq_loss.to(dtype=torch_module.float32)
                reconstructed_frames = inverse_preprocessor.inverse_preprocess_batch(
                    reconstructed_normalized_frames,
                    tensor_batch.side_means,
                    tensor_batch.side_log_sigmas,
                )
                task_loss_tensor = None
                if (
                    self.experiment_config.task is not None
                    and self.experiment_config.training.loss.task_weight > 0.0
                ):
                    if tensor_batch.noise_floors is None:
                        raise CodecConfigurationError(
                            "Task loss requested but the dataset batch does not "
                            "contain noise floors.",
                        )
                    assert task_frequency_grid_hz is not None
                    task_loss_tensor = torch_illustrative_task_loss(
                        tensor_batch.original_frames,
                        reconstructed_frames,
                        noise_floors=tensor_batch.noise_floors,
                        frequency_grid_hz=task_frequency_grid_hz,
                        config=self.experiment_config.task,
                    )
                total_loss, breakdown = compose_rate_distortion_loss(
                    reference_frames=tensor_batch.original_frames,
                    reconstructed_frames=reconstructed_frames,
                    rate_bits_per_frame=rate_bits,
                    side_information_bits=side_information_bits,
                    vq_loss=vq_loss,
                    dynamic_range_offset=self.experiment_config.runtime.preprocessing.dynamic_range_offset,
                    weights=self.experiment_config.training.loss,
                    task_loss=task_loss_tensor,
                )
                checked_tensors = {
                    "reconstructed_normalized_frames": reconstructed_normalized_frames,
                    "rate_bits": rate_bits,
                    "vq_loss": vq_loss,
                    "reconstructed_frames": reconstructed_frames,
                    "total_loss": total_loss,
                }
                if task_loss_tensor is not None:
                    checked_tensors["task_loss"] = task_loss_tensor
                _raise_if_non_finite_tensors(checked_tensors)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self._grad_scaler is not None:
                        self._grad_scaler.scale(total_loss).backward()
                        if self.experiment_config.training.gradient_clip_norm is not None:
                            self._grad_scaler.unscale_(self.optimizer)
                            torch_module.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.experiment_config.training.gradient_clip_norm,
                            )
                        self._grad_scaler.step(self.optimizer)
                        self._grad_scaler.update()
                    else:
                        total_loss.backward()  # type: ignore[no-untyped-call]
                        if self.experiment_config.training.gradient_clip_norm is not None:
                            torch_module.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.experiment_config.training.gradient_clip_norm,
                            )
                        self.optimizer.step()

            batch_size = tensor_batch.original_frames.shape[0]
            validation_diagnostics = None
            if not training:
                # Exact deployment-aligned diagnostics are intentionally validation-only.
                # Running them on training batches forces a GPU->CPU round-trip and
                # NumPy-side analysis on every optimizer step without affecting the loss.
                validation_diagnostics = self._compute_validation_diagnostics(
                    batch,
                    reconstructed_frames.detach().cpu().numpy(),
                    dataset_frequency_grid_hz,
                )
            aggregated.update(
                breakdown,
                batch_size=batch_size,
                validation_diagnostics=validation_diagnostics,
            )

        return aggregated.finalize()

    def _batch_to_tensors(self, batch: PreparedPsdBatch) -> _TorchBatch:
        """Move one NumPy batch onto the configured torch device."""
        noise_floors = None
        if batch.noise_floors is not None:
            noise_floors = self._numpy_batch_to_device_tensor(
                batch.noise_floors,
            )
        return _TorchBatch(
            original_frames=self._numpy_batch_to_device_tensor(batch.original_frames),
            normalized_frames=self._numpy_batch_to_device_tensor(batch.normalized_frames),
            side_means=self._numpy_batch_to_device_tensor(batch.side_means),
            side_log_sigmas=self._numpy_batch_to_device_tensor(batch.side_log_sigmas),
            noise_floors=noise_floors,
        )

    def _numpy_batch_to_device_tensor(
        self,
        values: np.ndarray,
    ) -> Tensor:
        """Convert one contiguous NumPy batch into a torch tensor on the training device."""
        torch_module = _require_torch()
        cpu_tensor = torch_module.from_numpy(
            np.asarray(values, dtype=np.float32, order="C")
        )
        if self._pin_memory and self._training_device_type == "cuda":
            cpu_tensor = cpu_tensor.pin_memory()
            return cast(
                Tensor,
                cpu_tensor.to(self._training_device_handle, non_blocking=True),
            )
        return cast(Tensor, cpu_tensor.to(self._training_device_handle))

    def _autocast_context(self) -> Any:
        """Return the resolved autocast context for the active training device."""
        torch_module = _require_torch()
        if not self._autocast_enabled:
            return contextlib.nullcontext()
        return cast(
            Any,
            torch_module.autocast(
                device_type=self._training_device_type,
                dtype=self._autocast_dtype,
            ),
        )

    def _compute_validation_diagnostics(
        self,
        batch: PreparedPsdBatch,
        reconstructed_frames: np.ndarray,
        frequency_grid_hz: np.ndarray | None,
    ) -> _ValidationDiagnostics | None:
        """Compute exact deployment-aligned validation metrics for one batch.

        Purpose:
            Validation must judge the trained codec against the same quantities shown
            in the deployment notebook, not only against differentiable surrogates.
            This helper therefore compares the learned reconstruction against the
            reference frames using exact PSD-task-adjacent diagnostics. The fixed
            preprocessing-only baseline is precomputed once per validation dataset.
        """
        if not self._validation_diagnostics_required:
            return None

        diagnostics = _ValidationDiagnostics(
            peak_frequency_error_hz=(
                None
                if frequency_grid_hz is None
                else _batch_peak_frequency_error_hz(
                    batch.original_frames,
                    reconstructed_frames,
                    frequency_grid_hz,
                )
            ),
            peak_power_error_db=_batch_peak_power_error_db(
                batch.original_frames,
                reconstructed_frames,
            ),
        )
        if self.experiment_config.task is None:
            return diagnostics
        if batch.noise_floors is None or frequency_grid_hz is None:
            return diagnostics

        task_values = [
            illustrative_task_loss(
                reference_frame=reference_frame,
                reconstructed_frame=reconstructed_frame,
                noise_floor=noise_floor,
                frequency_grid_hz=frequency_grid_hz,
                config=self.experiment_config.task,
            )
            for reference_frame, reconstructed_frame, noise_floor in zip(
                batch.original_frames,
                reconstructed_frames,
                batch.noise_floors,
                strict=True,
            )
        ]
        return _ValidationDiagnostics(
            task_monitor=float(np.mean(task_values)),
            peak_frequency_error_hz=diagnostics.peak_frequency_error_hz,
            peak_power_error_db=diagnostics.peak_power_error_db,
        )

    def _compute_validation_baseline_metrics(
        self,
        validation_dataset: PreparedPsdDataset,
        *,
        inverse_preprocessor: DifferentiableInversePreprocessor,
    ) -> _ValidationBaselineMetrics:
        """Precompute the fixed preprocessing-only validation diagnostics once.

        Purpose:
            The preprocessing-only reconstruction does not change across epochs, so
            recomputing its exact PSD and task metrics inside every validation pass
            wastes substantial CPU time. This helper amortizes that cost up front.
        """
        torch_module = _require_torch()
        preprocessing_frame_batches: list[np.ndarray] = []
        chunk_size = max(1, self.experiment_config.training.batch_size * 4)
        for start_index in range(0, len(validation_dataset), chunk_size):
            stop_index = min(len(validation_dataset), start_index + chunk_size)
            preprocessing_batch = inverse_preprocessor.inverse_preprocess_batch(
                torch_module.as_tensor(
                    validation_dataset.normalized_frames[start_index:stop_index],
                    dtype=torch_module.float32,
                ),
                torch_module.as_tensor(
                    validation_dataset.side_means[start_index:stop_index],
                    dtype=torch_module.float32,
                ),
                torch_module.as_tensor(
                    validation_dataset.side_log_sigmas[start_index:stop_index],
                    dtype=torch_module.float32,
                ),
            )
            preprocessing_frame_batches.append(preprocessing_batch.cpu().numpy())

        preprocessing_only_frames = np.concatenate(preprocessing_frame_batches, axis=0)
        if preprocessing_only_frames.shape != validation_dataset.original_frames.shape:
            raise CodecConfigurationError(
                "Preprocessing-only validation frames do not align with the validation dataset."
            )

        preprocessing_psd_loss = _batch_log_spectral_distortion(
            validation_dataset.original_frames,
            preprocessing_only_frames,
            dynamic_range_offset=self.experiment_config.runtime.preprocessing.dynamic_range_offset,
        )
        peak_frequency_error_hz = (
            None
            if validation_dataset.frequency_grid_hz is None
            else _batch_peak_frequency_error_hz(
                validation_dataset.original_frames,
                preprocessing_only_frames,
                validation_dataset.frequency_grid_hz,
            )
        )
        peak_power_error_db = _batch_peak_power_error_db(
            validation_dataset.original_frames,
            preprocessing_only_frames,
        )
        task_monitor = None
        if (
            self.experiment_config.task is not None
            and validation_dataset.noise_floors is not None
            and validation_dataset.frequency_grid_hz is not None
        ):
            task_values = [
                illustrative_task_loss(
                    reference_frame=reference_frame,
                    reconstructed_frame=preprocessing_frame,
                    noise_floor=noise_floor,
                    frequency_grid_hz=validation_dataset.frequency_grid_hz,
                    config=self.experiment_config.task,
                )
                for reference_frame, preprocessing_frame, noise_floor in zip(
                    validation_dataset.original_frames,
                    preprocessing_only_frames,
                    validation_dataset.noise_floors,
                    strict=True,
                )
            ]
            task_monitor = float(np.mean(task_values))
        return _ValidationBaselineMetrics(
            preprocessing_psd_loss=preprocessing_psd_loss,
            preprocessing_task_monitor=task_monitor,
            preprocessing_peak_frequency_error_hz=peak_frequency_error_hz,
            preprocessing_peak_power_error_db=peak_power_error_db,
        )

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        epoch_metrics: EpochMetrics,
        best_selection_score: float,
        best_validation_loss: float,
    ) -> None:
        """Persist the current model, optimizer, and configuration state."""
        torch_module = _require_torch()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch_module.save(
            {
                "epoch_index": epoch_metrics.epoch_index,
                "selection_metric": self.experiment_config.artifacts.selection_metric,
                "best_selection_score": best_selection_score,
                "best_validation_loss": best_validation_loss,
                "experiment_config": self.experiment_config.to_dict(),
                "metrics": asdict(epoch_metrics),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def _export_runtime_assets(self, runtime_asset_dir: Path) -> None:
        """Persist the codebook, entropy probabilities, and runtime configuration."""
        _write_runtime_assets_from_model(
            self.model,
            runtime_asset_dir,
            self.experiment_config.runtime,
        )


def load_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str = "cpu",
) -> LoadedTrainingCheckpoint:
    """Load a persisted training checkpoint into typed Python objects."""
    torch_module = _require_torch()
    payload = torch_module.load(
        Path(checkpoint_path), map_location=map_location, weights_only=False
    )
    experiment_config = TrainingExperimentConfig.from_dict(payload["experiment_config"])
    metrics = EpochMetrics(**payload["metrics"])
    return LoadedTrainingCheckpoint(
        epoch_index=int(payload["epoch_index"]),
        selection_metric=str(payload.get("selection_metric", "validation_loss")),
        best_selection_score=float(
            payload.get("best_selection_score", payload["best_validation_loss"])
        ),
        best_validation_loss=float(payload["best_validation_loss"]),
        experiment_config=experiment_config,
        metrics=metrics,
        model_state_dict=payload["model_state_dict"],
        optimizer_state_dict=payload["optimizer_state_dict"],
    )


def run_training_experiment(
    experiment_config: TrainingExperimentConfig,
    *,
    source_config_path: Path | None = None,
    progress_reporter: Callable[[EpochProgressUpdate], None] | None = None,
) -> TrainingSummary:
    """Load the configured dataset, train the model, and export artifacts."""
    trainer = TorchCodecTrainer(experiment_config)
    training_dataset, validation_dataset = trainer.load_prepared_datasets()
    return trainer.fit(
        training_dataset,
        validation_dataset,
        source_config_path=source_config_path,
        progress_reporter=progress_reporter,
    )


def resolve_accelerator_training_device_string(
    configured_device: str,  # Requested device string from the experiment config
) -> str:
    """Resolve a training device and require it to be a real accelerator.

    Purpose:
        Some entrypoints, such as the canonical demo-training CLI, should fail fast
        when `auto` would otherwise fall back to CPU. This helper keeps that policy
        explicit at the orchestration boundary while reusing the trainer's validated
        device-resolution logic.
    """
    resolved_device = _resolve_training_device_string(configured_device)
    resolved_device_type = _require_torch().device(resolved_device).type
    if resolved_device_type not in _ACCELERATOR_DEVICE_TYPES:
        raise CodecConfigurationError(
            "This training run requires an accelerator, but the resolved device is "
            f"'{resolved_device}'. Configure a usable CUDA or MPS device, or opt out "
            "of the accelerator requirement explicitly.",
        )
    return resolved_device


def _resolve_epoch_selection_score(
    epoch_metrics: EpochMetrics,
    *,
    selection_metric: str,
) -> float:
    """Return the scalar metric used to pick the best checkpoint for one epoch."""
    selected_value = getattr(epoch_metrics, selection_metric)
    if selected_value is None:
        raise CodecConfigurationError(
            f"selection_metric '{selection_metric}' is unavailable for this experiment.",
        )
    return float(selected_value)


@dataclass(frozen=True)
class _TorchBatch:
    """Torch tensor batch used internally by the trainer."""

    original_frames: Tensor
    normalized_frames: Tensor
    side_means: Tensor
    side_log_sigmas: Tensor
    noise_floors: Tensor | None


@dataclass
class _AggregatedMetrics:
    """Mutable accumulator used to average epoch metrics."""

    total_loss_sum: float = 0.0
    psd_loss_sum: float = 0.0
    rate_bits_sum: float = 0.0
    side_information_bits_sum: float = 0.0
    vq_loss_sum: float = 0.0
    task_loss_sum: float = 0.0
    sample_count: int = 0
    task_monitor_sum: float = 0.0
    task_monitor_count: int = 0
    peak_frequency_error_hz_sum: float = 0.0
    peak_frequency_error_hz_count: int = 0
    peak_power_error_db_sum: float = 0.0
    peak_power_error_db_count: int = 0

    def update(
        self,
        breakdown: TrainingLossBreakdown,
        *,
        batch_size: int,
        validation_diagnostics: _ValidationDiagnostics | None,
    ) -> None:
        """Accumulate one batch worth of scalar metrics."""
        self.total_loss_sum += breakdown.total_loss * batch_size
        self.psd_loss_sum += breakdown.psd_loss * batch_size
        self.rate_bits_sum += breakdown.rate_bits * batch_size
        self.side_information_bits_sum += breakdown.side_information_bits * batch_size
        self.vq_loss_sum += breakdown.vq_loss * batch_size
        self.task_loss_sum += breakdown.task_loss * batch_size
        self.sample_count += batch_size
        if validation_diagnostics is None:
            return

        if validation_diagnostics.task_monitor is not None:
            self.task_monitor_sum += validation_diagnostics.task_monitor * batch_size
            self.task_monitor_count += batch_size
        if validation_diagnostics.peak_frequency_error_hz is not None:
            self.peak_frequency_error_hz_sum += (
                validation_diagnostics.peak_frequency_error_hz * batch_size
            )
            self.peak_frequency_error_hz_count += batch_size
        if validation_diagnostics.peak_power_error_db is not None:
            self.peak_power_error_db_sum += (
                validation_diagnostics.peak_power_error_db * batch_size
            )
            self.peak_power_error_db_count += batch_size

    def finalize(self) -> _FinalizedMetrics:
        """Return averaged scalar metrics for one epoch."""
        if self.sample_count <= 0:
            raise CodecConfigurationError("Cannot finalize epoch metrics with zero samples.")
        task_monitor = None
        if self.task_monitor_count > 0:
            task_monitor = self.task_monitor_sum / self.task_monitor_count
        return _FinalizedMetrics(
            total_loss=self.total_loss_sum / self.sample_count,
            psd_loss=self.psd_loss_sum / self.sample_count,
            rate_bits=self.rate_bits_sum / self.sample_count,
            side_information_bits=self.side_information_bits_sum / self.sample_count,
            vq_loss=self.vq_loss_sum / self.sample_count,
            task_loss=self.task_loss_sum / self.sample_count,
            task_monitor=task_monitor,
            peak_frequency_error_hz=(
                None
                if self.peak_frequency_error_hz_count == 0
                else self.peak_frequency_error_hz_sum / self.peak_frequency_error_hz_count
            ),
            peak_power_error_db=(
                None
                if self.peak_power_error_db_count == 0
                else self.peak_power_error_db_sum / self.peak_power_error_db_count
            ),
        )


@dataclass(frozen=True)
class _FinalizedMetrics:
    """Averaged scalar metrics returned by one epoch run."""

    total_loss: float
    psd_loss: float
    rate_bits: float
    side_information_bits: float
    vq_loss: float
    task_loss: float
    task_monitor: float | None = None
    peak_frequency_error_hz: float | None = None
    peak_power_error_db: float | None = None


@dataclass(frozen=True)
class _ValidationBaselineMetrics:
    """Exact preprocessing-only diagnostics that stay fixed across epochs."""

    preprocessing_psd_loss: float
    preprocessing_task_monitor: float | None = None
    preprocessing_peak_frequency_error_hz: float | None = None
    preprocessing_peak_power_error_db: float | None = None


@dataclass(frozen=True)
class _ValidationDiagnostics:
    """Exact validation-only diagnostics aligned with deployment analysis."""

    task_monitor: float | None = None
    peak_frequency_error_hz: float | None = None
    peak_power_error_db: float | None = None


def _batch_log_spectral_distortion(
    reference_frames: np.ndarray,
    candidate_frames: np.ndarray,
    *,
    dynamic_range_offset: float,
) -> float:
    """Return the mean PSD distortion over one NumPy validation batch."""
    difference = np.log(reference_frames + dynamic_range_offset) - np.log(
        candidate_frames + dynamic_range_offset
    )
    per_frame = np.mean(difference * difference, axis=1)
    return float(np.mean(per_frame))


def _batch_peak_frequency_error_hz(
    reference_frames: np.ndarray,
    candidate_frames: np.ndarray,
    frequency_grid_hz: np.ndarray,
) -> float:
    """Return the mean raw dominant-peak location error over one batch."""
    reference_indices = np.argmax(reference_frames, axis=1)
    candidate_indices = np.argmax(candidate_frames, axis=1)
    frequency_grid = np.asarray(frequency_grid_hz, dtype=np.float64)
    errors = np.abs(frequency_grid[reference_indices] - frequency_grid[candidate_indices])
    return float(np.mean(errors))


def _batch_peak_power_error_db(
    reference_frames: np.ndarray,
    candidate_frames: np.ndarray,
) -> float:
    """Return the mean dominant-peak amplitude error over one batch."""
    reference_db = 10.0 * np.log10(np.maximum(np.max(reference_frames, axis=1), 1.0e-12))
    candidate_db = 10.0 * np.log10(np.maximum(np.max(candidate_frames, axis=1), 1.0e-12))
    return float(np.mean(np.abs(reference_db - candidate_db)))


def _compose_validation_deployment_score(
    *,
    validation_psd_loss: float,
    validation_preprocessing_psd_loss: float | None,
    validation_peak_frequency_error_hz: float | None,
    validation_preprocessing_peak_frequency_error_hz: float | None,
    validation_peak_power_error_db: float | None,
    validation_preprocessing_peak_power_error_db: float | None,
    validation_task_monitor: float | None,
    validation_preprocessing_task_monitor: float | None,
) -> float | None:
    """Return a preprocessing-relative deployment score for checkpoint selection.

    A score of `1.0` means parity with the deterministic preprocessing-only baseline.
    Lower values are better. The components are weighted toward raw dominant-peak
    localization because that failure mode dominated the deployment notebook results.
    """
    weighted_sum = 0.0
    total_weight = 0.0

    def add_component(
        *,
        key: str,
        candidate_value: float | None,
        baseline_value: float | None,
    ) -> None:
        nonlocal weighted_sum, total_weight
        if candidate_value is None or baseline_value is None:
            return
        stabilizer = _DEPLOYMENT_SELECTION_COMPONENT_STABILIZERS[key]
        component_weight = _DEPLOYMENT_SELECTION_COMPONENT_WEIGHTS[key]
        weighted_sum += component_weight * (
            (candidate_value + stabilizer) / (baseline_value + stabilizer)
        )
        total_weight += component_weight

    add_component(
        key="psd",
        candidate_value=validation_psd_loss,
        baseline_value=validation_preprocessing_psd_loss,
    )
    add_component(
        key="peak_frequency",
        candidate_value=validation_peak_frequency_error_hz,
        baseline_value=validation_preprocessing_peak_frequency_error_hz,
    )
    add_component(
        key="peak_power",
        candidate_value=validation_peak_power_error_db,
        baseline_value=validation_preprocessing_peak_power_error_db,
    )
    add_component(
        key="task",
        candidate_value=validation_task_monitor,
        baseline_value=validation_preprocessing_task_monitor,
    )
    if total_weight <= 0.0:
        return None
    return weighted_sum / total_weight


def _selection_candidate_is_acceptable(
    epoch_metrics: EpochMetrics,
    *,
    require_selection_to_beat_preprocessing: bool,
) -> bool:
    """Return whether one epoch may become the persisted best checkpoint."""
    if not require_selection_to_beat_preprocessing:
        return True
    deployment_score = epoch_metrics.validation_deployment_score
    if deployment_score is None:
        raise CodecConfigurationError(
            "require_selection_to_beat_preprocessing needs validation_deployment_score, "
            "but the configured experiment does not produce it."
        )
    return deployment_score < 1.0


def _coerce_mapping(value: object) -> dict[str, Any]:
    """Return a typed mapping or raise a precise configuration error."""
    if not isinstance(value, dict):
        raise CodecConfigurationError("Expected a mapping in the experiment configuration.")
    return dict(value)


def _resolve_training_device_string(
    configured_device: str,  # Requested device string from the experiment config
) -> str:
    """Resolve the effective training device and verify that it is usable.

    Purpose:
        Make device selection self-contained and robust. `auto` should prefer the
        first actually usable accelerator, not merely one that reports itself as
        nominally available. Explicit device requests are also validated eagerly so
        training fails fast with a precise configuration error instead of a later
        runtime/backend exception.
    """
    normalized_device = configured_device.strip()
    if normalized_device == "auto":
        for candidate_device in ("cuda", "mps", "cpu"):
            if _can_use_training_device(candidate_device):
                return candidate_device
        raise CodecConfigurationError(
            "Unable to resolve a usable training device from the auto candidates.",
        )
    if not _can_use_training_device(normalized_device):
        raise CodecConfigurationError(
            f"Requested training device '{normalized_device}' is not usable on this system.",
        )
    return normalized_device


def _can_use_training_device(
    device_name: str,  # Candidate torch device string such as `cuda` or `cpu`
) -> bool:
    """Return whether the requested training device is genuinely usable.

    The check is intentionally stronger than an availability flag: it validates the
    backend-specific capability signal and then performs a tiny allocation/probe on
    the target device so the trainer can rely on the returned device string.
    """
    torch_module = _require_torch()
    try:
        device = torch_module.device(device_name)
    except (RuntimeError, TypeError, ValueError):
        return False

    if device.type == "cpu":
        return True
    if device.type == "cuda":
        if not torch_module.cuda.is_available():
            return False
        if device.index is not None and device.index >= torch_module.cuda.device_count():
            return False
    elif device.type == "mps":
        mps_backend = getattr(torch_module.backends, "mps", None)
        if mps_backend is None or not bool(mps_backend.is_available()):
            return False

    try:
        probe = torch_module.zeros(1, dtype=torch_module.float32, device=device)
        _ = probe + 1.0
        if device.type == "cuda":
            torch_module.cuda.synchronize(device)
    except Exception:
        return False
    return True


def _resolve_autocast_dtype(
    mixed_precision: str,
    *,
    device: Any,
) -> Any | None:
    """Resolve the autocast dtype for the active training device.

    Purpose:
        Keep the model matmuls in reduced precision on CUDA when that is safe, while
        leaving numerically sensitive reconstruction and loss code in explicit
        float32 outside autocast.
    """
    torch_module = _require_torch()
    if mixed_precision == "disabled":
        return None
    if device.type != "cuda":
        return None
    if mixed_precision == "auto":
        return (
            torch_module.bfloat16
            if torch_module.cuda.is_bf16_supported()
            else torch_module.float16
        )
    if mixed_precision == "bf16":
        if not torch_module.cuda.is_bf16_supported():
            raise CodecConfigurationError(
                "mixed_precision='bf16' requires CUDA bfloat16 support on this system."
            )
        return torch_module.bfloat16
    if mixed_precision == "fp16":
        return torch_module.float16
    raise CodecConfigurationError(f"Unsupported mixed_precision mode: {mixed_precision}.")


def _resolve_pin_memory_enabled(
    configured_pin_memory: bool | None,
    *,
    device_type: str,
) -> bool:
    """Return whether host batches should be pinned before GPU transfer."""
    if configured_pin_memory is not None:
        return configured_pin_memory
    return device_type == "cuda"


def _resolve_data_loader_worker_count(
    configured_worker_count: int | None,
    *,
    device_type: str,
) -> int:
    """Return the DataLoader worker count for one training run.

    Purpose:
        CUDA runs benefit from overlapping host-side collation with GPU execution,
        while CPU-only runs often lose time to multiprocessing overhead on in-memory
        NumPy datasets. The auto policy therefore stays conservative on CPU.
    """
    if configured_worker_count is not None:
        return configured_worker_count
    if device_type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 1
    return max(2, min(8, cpu_count // 2))


def _requires_exact_validation_diagnostics(
    experiment_config: TrainingExperimentConfig,
) -> bool:
    """Return whether one experiment needs exact deployment-style validation metrics."""
    if experiment_config.artifacts.selection_metric != "validation_loss":
        return True
    if experiment_config.artifacts.require_selection_to_beat_preprocessing:
        return True
    return experiment_config.task is not None


def _expect_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required nested mapping from a configuration payload."""
    if key not in payload:
        raise CodecConfigurationError(f"Missing required configuration section '{key}'.")
    return _coerce_mapping(payload[key])


def _coerce_path_fields(
    payload: dict[str, Any],
    path_keys: set[str],
) -> dict[str, Any]:
    """Convert selected string fields into `Path` objects."""
    converted = dict(payload)
    for key in path_keys:
        if key in converted:
            converted[key] = Path(str(converted[key]))
    return converted


def _parse_runtime_config(payload: dict[str, Any]) -> CodecRuntimeConfig:
    """Parse the nested runtime configuration subtree."""
    preprocessing_payload = _expect_mapping(payload, "preprocessing")
    entropy_payload = _expect_mapping(payload, "entropy_model")
    packet_payload = _coerce_mapping(payload.get("packet_format", {}))
    if "magic" in packet_payload:
        packet_payload["magic"] = str(packet_payload["magic"]).encode("ascii")
    mean_quantizer_payload = _expect_mapping(preprocessing_payload, "mean_quantizer")
    log_sigma_quantizer_payload = _expect_mapping(preprocessing_payload, "log_sigma_quantizer")
    preprocessing_fields = dict(preprocessing_payload)
    preprocessing_fields["mean_quantizer"] = ScalarQuantizerConfig(**mean_quantizer_payload)
    preprocessing_fields["log_sigma_quantizer"] = ScalarQuantizerConfig(
        **log_sigma_quantizer_payload
    )
    return CodecRuntimeConfig(
        preprocessing=PreprocessingConfig(**preprocessing_fields),
        entropy_model=FactorizedEntropyModelConfig(**entropy_payload),
        packet_format=PacketFormatConfig(**packet_payload)
        if packet_payload
        else PacketFormatConfig(),
    )


def _parse_training_config(payload: dict[str, Any]) -> TrainingConfig:
    """Parse the nested training configuration subtree."""
    training_fields = dict(payload)
    loss_payload = _coerce_mapping(training_fields.pop("loss", {}))
    return TrainingConfig(
        **training_fields,
        loss=RateDistortionLossConfig(**loss_payload),
    )


def _runtime_config_to_dict(config: CodecRuntimeConfig) -> dict[str, Any]:
    """Serialize a runtime configuration into a JSON-safe mapping."""
    return {
        "preprocessing": {
            **_dataclass_to_dict(config.preprocessing),
            "mean_quantizer": _dataclass_to_dict(config.preprocessing.mean_quantizer),
            "log_sigma_quantizer": _dataclass_to_dict(config.preprocessing.log_sigma_quantizer),
        },
        "entropy_model": _dataclass_to_dict(config.entropy_model),
        "packet_format": {
            **_dataclass_to_dict(config.packet_format),
            "magic": config.packet_format.magic.decode("ascii"),
        },
    }


def _dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Convert a dataclass to a JSON/YAML-safe dictionary."""
    result = asdict(value)
    for key, field_value in list(result.items()):
        if isinstance(field_value, Path):
            result[key] = str(field_value)
    return result
