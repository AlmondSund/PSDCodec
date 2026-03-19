"""Training, checkpointing, and export orchestration for the PyTorch codec."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def _require_torch() -> Any:
    """Return the imported torch module or raise a precise error."""
    if _torch is None or TorchDataLoader is None:
        raise ImportError("PyTorch is required to use pipelines.training.")
    return _torch


def _raise_if_non_finite_tensor(
    value: Tensor,  # Tensor whose entries must all be finite
    *,
    name: str,
) -> None:
    """Raise a precise floating-point error when a tensor contains NaN or Inf.

    Purpose:
        Surface numerical instability at the training boundary where it first becomes
        observable, instead of letting non-finite tensors fail later inside monitoring
        or serialization code with a less actionable traceback.
    """
    torch_module = _require_torch()
    if not bool(torch_module.isfinite(value).all().item()):
        raise FloatingPointError(
            f"{name} contains non-finite values. This indicates numerical instability "
            "in the training path before exact task monitoring.",
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
    device: str = "cpu"  # Torch device string or `auto`
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


@dataclass(frozen=True)
class ArtifactConfig:
    """Output locations and export switches for one experiment."""

    experiment_name: str  # Stable artifact namespace under models/checkpoints and models/exports
    checkpoint_root: Path = Path("models/checkpoints")  # Root directory for checkpoints
    export_root: Path = Path("models/exports")  # Root directory for export-ready artifacts
    export_onnx: bool = True  # Whether to export the encoder boundary to ONNX after training
    save_latest_checkpoint: bool = True  # Whether to persist the latest checkpoint every epoch
    save_best_checkpoint: bool = True  # Whether to persist the best validation checkpoint


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


@dataclass(frozen=True)
class TrainingSummary:
    """Training outputs and persisted artifact locations."""

    history: tuple[EpochMetrics, ...]
    best_epoch_index: int
    best_validation_loss: float
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
    best_validation_loss: float  # Best validation loss observed so far


@dataclass(frozen=True)
class LoadedTrainingCheckpoint:
    """Typed view over a persisted training checkpoint."""

    epoch_index: int
    best_validation_loss: float
    experiment_config: TrainingExperimentConfig
    metrics: EpochMetrics
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]


class TorchCodecTrainer:
    """Application-layer trainer for the PyTorch PSD codec."""

    def __init__(self, experiment_config: TrainingExperimentConfig) -> None:
        """Initialize model, optimizer, and deterministic preprocessing helpers."""
        torch_module = _require_torch()
        self.experiment_config = experiment_config
        self.preprocessor = FramePreprocessor(experiment_config.runtime.preprocessing)
        self.training_device = _resolve_training_device_string(experiment_config.training.device)
        self.model = TorchFullCodec(experiment_config.model).to(self.training_device)
        self.optimizer = torch_module.optim.Adam(
            self.model.parameters(),
            lr=experiment_config.training.learning_rate,
            weight_decay=experiment_config.training.weight_decay,
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

        train_loader: Any = TorchDataLoader(
            training_dataset,
            batch_size=self.experiment_config.training.batch_size,
            shuffle=self.experiment_config.dataset.shuffle,
            collate_fn=collate_prepared_psd_samples,
        )
        validation_loader: Any = TorchDataLoader(
            validation_dataset,
            batch_size=self.experiment_config.training.batch_size,
            shuffle=False,
            collate_fn=collate_prepared_psd_samples,
        )
        side_information_bits = float(
            self.experiment_config.runtime.preprocessing.block_count
            * self.experiment_config.runtime.preprocessing.side_information_bits_per_block
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
            shutil.copy2(source_config_path, export_dir / source_config_path.name)

        history: list[EpochMetrics] = []
        best_validation_loss = float("inf")
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
            )
            history.append(epoch_metrics)

            latest_checkpoint_path = checkpoint_dir / "latest.pt"
            if self.experiment_config.artifacts.save_latest_checkpoint:
                self._save_checkpoint(
                    latest_checkpoint_path,
                    epoch_metrics=epoch_metrics,
                    best_validation_loss=min(best_validation_loss, epoch_metrics.validation_loss),
                )

            if epoch_metrics.validation_loss < best_validation_loss:
                best_validation_loss = epoch_metrics.validation_loss
                best_epoch_index = epoch_index
                best_model_state_dict = copy.deepcopy(self.model.state_dict())
                if self.experiment_config.artifacts.save_best_checkpoint:
                    best_checkpoint_path = checkpoint_dir / "best.pt"
                    self._save_checkpoint(
                        best_checkpoint_path,
                        epoch_metrics=epoch_metrics,
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
                        best_validation_loss=best_validation_loss,
                    )
                )

        if best_model_state_dict is not None:
            self.model.load_state_dict(best_model_state_dict)
        self._export_runtime_assets(runtime_asset_dir)
        onnx_path = None
        if self.experiment_config.artifacts.export_onnx:
            onnx_path = export_dir / "encoder.onnx"
            self.model.export_encoder_to_onnx(onnx_path)
        metadata_path = export_dir / "training_summary.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "experiment_config": self.experiment_config.to_dict(),
                    "best_epoch_index": best_epoch_index,
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
        return TrainingSummary(
            history=tuple(history),
            best_epoch_index=best_epoch_index,
            best_validation_loss=best_validation_loss,
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
                output = self.model(tensor_batch.normalized_frames)
                reconstructed_frames = inverse_preprocessor.inverse_preprocess_batch(
                    output.reconstructed_normalized_frames,
                    tensor_batch.side_means,
                    tensor_batch.side_log_sigmas,
                )
                _raise_if_non_finite_tensor(
                    reconstructed_frames,
                    name="reconstructed_frames",
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
                    rate_bits_per_frame=output.rate_bits,
                    side_information_bits=side_information_bits,
                    vq_loss=output.vq_loss,
                    dynamic_range_offset=self.experiment_config.runtime.preprocessing.dynamic_range_offset,
                    weights=self.experiment_config.training.loss,
                    task_loss=task_loss_tensor,
                )
                _raise_if_non_finite_tensor(total_loss, name="total_loss")

                if training:
                    self.optimizer.zero_grad()
                    total_loss.backward()  # type: ignore[no-untyped-call]
                    if self.experiment_config.training.gradient_clip_norm is not None:
                        torch_module.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.experiment_config.training.gradient_clip_norm,
                        )
                    self.optimizer.step()

            batch_size = tensor_batch.original_frames.shape[0]
            task_monitor = self._compute_validation_task_monitor(
                batch,
                reconstructed_frames.detach().cpu().numpy(),
                dataset_frequency_grid_hz,
            )
            aggregated.update(breakdown, batch_size=batch_size, task_monitor=task_monitor)

        return aggregated.finalize()

    def _batch_to_tensors(self, batch: PreparedPsdBatch) -> _TorchBatch:
        """Move one NumPy batch onto the configured torch device."""
        torch_module = _require_torch()
        device = torch_module.device(self.training_device)
        noise_floors = None
        if batch.noise_floors is not None:
            noise_floors = torch_module.as_tensor(
                batch.noise_floors, dtype=torch_module.float32, device=device
            )
        return _TorchBatch(
            original_frames=torch_module.as_tensor(
                batch.original_frames,
                dtype=torch_module.float32,
                device=device,
            ),
            normalized_frames=torch_module.as_tensor(
                batch.normalized_frames,
                dtype=torch_module.float32,
                device=device,
            ),
            side_means=torch_module.as_tensor(
                batch.side_means,
                dtype=torch_module.float32,
                device=device,
            ),
            side_log_sigmas=torch_module.as_tensor(
                batch.side_log_sigmas,
                dtype=torch_module.float32,
                device=device,
            ),
            noise_floors=noise_floors,
        )

    def _compute_validation_task_monitor(
        self,
        batch: PreparedPsdBatch,
        reconstructed_frames: np.ndarray,
        frequency_grid_hz: np.ndarray | None,
    ) -> float | None:
        """Compute the exact illustrative task metric for validation monitoring."""
        if self.experiment_config.task is None:
            return None
        if batch.noise_floors is None or frequency_grid_hz is None:
            return None
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
        return float(np.mean(task_values))

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        epoch_metrics: EpochMetrics,
        best_validation_loss: float,
    ) -> None:
        """Persist the current model, optimizer, and configuration state."""
        torch_module = _require_torch()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch_module.save(
            {
                "epoch_index": epoch_metrics.epoch_index,
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
        runtime_asset_dir.mkdir(parents=True, exist_ok=True)
        np.save(runtime_asset_dir / "codebook.npy", self.model.export_runtime_codebook())
        np.save(
            runtime_asset_dir / "entropy_probabilities.npy",
            self.model.export_runtime_probabilities(),
        )
        (runtime_asset_dir / "runtime_config.json").write_text(
            json.dumps(_runtime_config_to_dict(self.experiment_config.runtime), indent=2),
            encoding="utf-8",
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

    def update(
        self,
        breakdown: TrainingLossBreakdown,
        *,
        batch_size: int,
        task_monitor: float | None,
    ) -> None:
        """Accumulate one batch worth of scalar metrics."""
        self.total_loss_sum += breakdown.total_loss * batch_size
        self.psd_loss_sum += breakdown.psd_loss * batch_size
        self.rate_bits_sum += breakdown.rate_bits * batch_size
        self.side_information_bits_sum += breakdown.side_information_bits * batch_size
        self.vq_loss_sum += breakdown.vq_loss * batch_size
        self.task_loss_sum += breakdown.task_loss * batch_size
        self.sample_count += batch_size
        if task_monitor is not None:
            self.task_monitor_sum += task_monitor * batch_size
            self.task_monitor_count += batch_size

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


def _coerce_mapping(value: object) -> dict[str, Any]:
    """Return a typed mapping or raise a precise configuration error."""
    if not isinstance(value, dict):
        raise CodecConfigurationError("Expected a mapping in the experiment configuration.")
    return dict(value)


def _resolve_training_device_string(
    configured_device: str,  # Requested device string from the experiment config
) -> str:
    """Resolve `auto` device selection to a concrete torch device string."""
    torch_module = _require_torch()
    if configured_device != "auto":
        return configured_device
    if torch_module.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and bool(mps_backend.is_available()):
        return "mps"
    return "cpu"


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
