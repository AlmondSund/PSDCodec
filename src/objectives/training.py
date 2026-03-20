"""Torch-compatible training losses for PSDCodec rate-distortion optimization.

The exact manuscript task metrics live in :mod:`objectives.distortion` for runtime
evaluation and notebook analysis. This module provides the differentiable training
surrogates needed to optimize the same conceptual objective with backpropagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from codec.exceptions import CodecConfigurationError
from objectives.distortion import IllustrativeTaskConfig

_torch: Any | None
try:
    import torch as _torch
except ImportError:  # pragma: no cover - exercised only when torch is unavailable
    _torch = None

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

_PEAK_SOFTMAX_TEMPERATURE: float = 0.10


def _require_torch() -> Any:
    """Return the imported torch module or raise a precise error."""
    if _torch is None:
        raise ImportError("PyTorch is required to use objectives.training.")
    return _torch


@dataclass(frozen=True)
class RateDistortionLossConfig:
    """Loss weights for the Lagrangian training objective."""

    psd_weight: float = 1.0  # β_psd
    rate_weight: float = 1.0e-2  # λ_R
    vq_weight: float = 1.0  # η
    task_weight: float = 0.0  # β_task

    def __post_init__(self) -> None:
        """Validate the rate-distortion loss weights."""
        if self.psd_weight < 0.0:
            raise CodecConfigurationError("psd_weight must be non-negative.")
        if self.rate_weight < 0.0:
            raise CodecConfigurationError("rate_weight must be non-negative.")
        if self.vq_weight < 0.0:
            raise CodecConfigurationError("vq_weight must be non-negative.")
        if self.task_weight < 0.0:
            raise CodecConfigurationError("task_weight must be non-negative.")


@dataclass(frozen=True)
class TrainingLossBreakdown:
    """Scalar loss components reported for one optimization step or epoch."""

    total_loss: float
    psd_loss: float
    rate_bits: float
    side_information_bits: float
    vq_loss: float
    task_loss: float


def _detach_scalar_values_to_host(
    *values: Tensor,
) -> tuple[float, ...]:
    """Return scalar tensor values as Python floats using one host synchronization.

    Purpose:
        Training reports several scalar loss components per batch. Pulling each
        component to the host with its own `.item()` call forces repeated device
        synchronizations on CUDA runs. This helper stacks the scalars first so the
        hot path pays only one host transfer per batch.
    """
    torch_module = _require_torch()
    detached_scalars = torch_module.stack(
        [value.detach().reshape(()) for value in values]
    )
    host_values = detached_scalars.to(device="cpu", dtype=torch_module.float64)
    return tuple(float(value) for value in host_values.tolist())


@dataclass(frozen=True)
class TorchIllustrativeFeatureBatch:
    """Differentiable illustrative-feature surrogate evaluated over one batch."""

    peak_frequency_hz: Tensor  # Soft-argmax dominant-peak location surrogate
    peak_power_db: Tensor  # Soft dominant-peak power surrogate expressed in dB
    spectral_centroid_hz: Tensor  # Exact spectral centroid
    occupied_bandwidth_hz: Tensor  # Soft occupied-bandwidth surrogate


def torch_log_spectral_distortion(
    reference_frames: Tensor,
    reconstructed_frames: Tensor,
    *,
    dynamic_range_offset: float,
) -> Tensor:
    """Compute the manuscript PSD distortion over one batch."""
    torch_module = _require_torch()
    if dynamic_range_offset <= 0.0:
        raise CodecConfigurationError("dynamic_range_offset must be strictly positive.")
    difference = torch_module.log(reference_frames + dynamic_range_offset) - torch_module.log(
        reconstructed_frames + dynamic_range_offset,
    )
    return cast(Tensor, torch_module.mean(difference * difference))


def torch_occupancy_task_loss(
    reference_frames: Tensor,
    reconstructed_frames: Tensor,
    *,
    noise_floors: Tensor,
    config: IllustrativeTaskConfig,
) -> Tensor:
    """Compute a differentiable occupancy-consistency regularizer."""
    torch_module = _require_torch()
    reference_soft, reconstructed_soft = _torch_task_soft_occupancies(
        reference_frames,
        reconstructed_frames,
        noise_floors=noise_floors,
        config=config,
    )
    epsilon = max(float(torch_module.finfo(reconstructed_soft.dtype).eps), 1.0e-6)
    clipped = torch_module.clamp(reconstructed_soft, min=epsilon, max=1.0 - epsilon)
    positive = config.occupancy_positive_weight * reference_soft * torch_module.log(clipped)
    negative = (
        config.occupancy_negative_weight * (1.0 - reference_soft) * torch_module.log(1.0 - clipped)
    )
    return cast(Tensor, -torch_module.mean(positive + negative))


def torch_illustrative_task_loss(
    reference_frames: Tensor,
    reconstructed_frames: Tensor,
    *,
    noise_floors: Tensor,
    frequency_grid_hz: Tensor,
    config: IllustrativeTaskConfig,
) -> Tensor:
    """Compute the full differentiable illustrative-task surrogate for training.

    Purpose:
        Align the training objective with the manuscript's illustrative sensing task:
        occupancy consistency plus feature preservation on peak location, dominant
        peak power, spectral centroid, and occupied bandwidth.

    Notes:
        The exact manuscript features involve `argmax` and connected-component
        selection, which are not directly differentiable. This function therefore
        uses smooth surrogates that preserve the same semantic targets:

        - occupancy uses the exact soft thresholding term from the paper,
        - dominant peak location uses a normalized soft-argmax after smoothing,
        - dominant peak power uses the same soft peak weights applied to the raw frame,
        - spectral centroid is exact and differentiable,
        - occupied bandwidth uses a soft occupancy-weighted interval-width surrogate.
    """
    _validate_task_batch_inputs(
        reference_frames,
        reconstructed_frames,
        noise_floors=noise_floors,
        frequency_grid_hz=frequency_grid_hz,
    )
    reference_soft, reconstructed_soft = _torch_task_soft_occupancies(
        reference_frames,
        reconstructed_frames,
        noise_floors=noise_floors,
        config=config,
    )
    occupancy_loss = torch_occupancy_task_loss(
        reference_frames,
        reconstructed_frames,
        noise_floors=noise_floors,
        config=config,
    )
    reference_features = _torch_extract_illustrative_features(
        reference_frames,
        soft_occupancies=reference_soft,
        frequency_grid_hz=frequency_grid_hz,
        config=config,
    )
    reconstructed_features = _torch_extract_illustrative_features(
        reconstructed_frames,
        soft_occupancies=reconstructed_soft,
        frequency_grid_hz=frequency_grid_hz,
        config=config,
    )
    feature_loss = _torch_feature_preservation_loss(
        reference_features,
        reconstructed_features,
        config=config,
    )
    return config.occupancy_weight * occupancy_loss + config.feature_weight * feature_loss


def compose_rate_distortion_loss(
    *,
    reference_frames: Tensor,
    reconstructed_frames: Tensor,
    rate_bits_per_frame: Tensor,
    side_information_bits: float,
    vq_loss: Tensor,
    dynamic_range_offset: float,
    weights: RateDistortionLossConfig,
    task_loss: Tensor | None = None,
) -> tuple[Tensor, TrainingLossBreakdown]:
    """Compose the weighted training objective and return detached scalar metrics.

    Purpose:
        The returned tensor drives backpropagation, while the scalar breakdown feeds
        epoch-level logging and checkpoint-selection metrics. The reporting path is
        intentionally detached from autograd and uses a single host transfer for the
        tensor-valued scalars so accelerator runs do not stall on repeated `.item()`
        synchronizations.
    """
    torch_module = _require_torch()
    psd_loss = torch_log_spectral_distortion(
        reference_frames,
        reconstructed_frames,
        dynamic_range_offset=dynamic_range_offset,
    )
    average_rate_bits = torch_module.mean(rate_bits_per_frame)
    average_side_information_bits = float(side_information_bits)
    total_task_loss = (
        torch_module.zeros((), dtype=psd_loss.dtype, device=psd_loss.device)
        if task_loss is None
        else task_loss
    )
    total_loss = (
        weights.psd_weight * psd_loss
        + weights.rate_weight * (average_rate_bits + average_side_information_bits)
        + weights.vq_weight * vq_loss
        + weights.task_weight * total_task_loss
    )
    (
        total_loss_scalar,
        psd_loss_scalar,
        rate_bits_scalar,
        vq_loss_scalar,
        task_loss_scalar,
    ) = _detach_scalar_values_to_host(
        total_loss,
        psd_loss,
        average_rate_bits,
        vq_loss,
        total_task_loss,
    )
    return total_loss, TrainingLossBreakdown(
        total_loss=total_loss_scalar,
        psd_loss=psd_loss_scalar,
        rate_bits=rate_bits_scalar,
        side_information_bits=average_side_information_bits,
        vq_loss=vq_loss_scalar,
        task_loss=task_loss_scalar,
    )


def _validate_task_batch_inputs(
    reference_frames: Tensor,
    reconstructed_frames: Tensor,
    *,
    noise_floors: Tensor,
    frequency_grid_hz: Tensor,
) -> None:
    """Validate the aligned batched tensors needed by the illustrative-task surrogate."""
    if reference_frames.ndim != 2 or reconstructed_frames.ndim != 2:
        raise CodecConfigurationError(
            "reference_frames and reconstructed_frames must have shape [batch, bin_count]."
        )
    if tuple(reference_frames.shape) != tuple(reconstructed_frames.shape):
        raise CodecConfigurationError(
            "reference_frames and reconstructed_frames must have the same shape."
        )
    if tuple(noise_floors.shape) != tuple(reference_frames.shape):
        raise CodecConfigurationError("noise_floors must have the same shape as reference_frames.")
    if frequency_grid_hz.ndim != 1 or frequency_grid_hz.shape[0] != reference_frames.shape[1]:
        raise CodecConfigurationError(
            "frequency_grid_hz must have shape [bin_count] matching the frame width."
        )


def _torch_task_soft_occupancies(
    reference_frames: Tensor,
    reconstructed_frames: Tensor,
    *,
    noise_floors: Tensor,
    config: IllustrativeTaskConfig,
) -> tuple[Tensor, Tensor]:
    """Return the batched soft occupancies for the illustrative task."""
    torch_module = _require_torch()
    logits_reference = (
        reference_frames - noise_floors - config.occupancy_margin
    ) / config.occupancy_temperature
    logits_reconstructed = (
        reconstructed_frames - noise_floors - config.occupancy_margin
    ) / config.occupancy_temperature
    return (
        cast(Tensor, torch_module.sigmoid(logits_reference)),
        cast(Tensor, torch_module.sigmoid(logits_reconstructed)),
    )


def _torch_extract_illustrative_features(
    frames: Tensor,
    *,
    soft_occupancies: Tensor,
    frequency_grid_hz: Tensor,
    config: IllustrativeTaskConfig,
) -> TorchIllustrativeFeatureBatch:
    """Extract differentiable illustrative features from one batched PSD tensor."""
    torch_module = _require_torch()
    frequency_grid = frequency_grid_hz.unsqueeze(0)
    bandwidth_stabilizer_hz = _torch_frequency_grid_step_hz(frequency_grid_hz)

    # The manuscript peak uses a hard argmax after smoothing. Training replaces it
    # with a normalized soft-argmax so gradients still encourage dominant-peak
    # alignment without changing the target quantity being approximated.
    smoothed_frames = _torch_moving_average(
        frames,
        window_length=config.smoothing_window_bins,
    )
    peak_normalizer = torch_module.clamp(
        torch_module.amax(smoothed_frames, dim=1, keepdim=True),
        min=1.0e-12,
    )
    peak_logits = (smoothed_frames / peak_normalizer) / _PEAK_SOFTMAX_TEMPERATURE
    peak_weights = torch_module.softmax(peak_logits, dim=1)
    peak_frequency_hz = torch_module.sum(peak_weights * frequency_grid, dim=1)

    # Use the same soft dominant-peak assignment to measure raw peak amplitude in dB.
    # This makes the training surrogate explicitly care about the dominant-peak power
    # that the deployment score already penalizes.
    peak_power_linear = torch_module.sum(peak_weights * frames, dim=1)
    peak_power_db = 10.0 * torch_module.log10(torch_module.clamp(peak_power_linear, min=1.0e-12))

    frame_mass = torch_module.sum(frames, dim=1)
    safe_frame_mass = torch_module.clamp(frame_mass, min=1.0e-12)
    centroid_hz = torch_module.sum(frames * frequency_grid, dim=1) / safe_frame_mass
    centroid_hz = torch_module.where(
        frame_mass > 1.0e-12,
        centroid_hz,
        torch_module.zeros_like(frame_mass),
    )

    # The exact manuscript bandwidth selects the strongest connected occupied
    # component. The differentiable surrogate models occupied support as a soft
    # interval whose width matches the variance of the soft occupancy-weighted mass.
    occupied_weights = frames * soft_occupancies
    occupied_mass = torch_module.sum(occupied_weights, dim=1)
    safe_occupied_mass = torch_module.clamp(occupied_mass, min=1.0e-12)
    occupied_center_hz = (
        torch_module.sum(occupied_weights * frequency_grid, dim=1) / safe_occupied_mass
    )
    occupied_variance_hz2 = torch_module.sum(
        occupied_weights * (frequency_grid - occupied_center_hz.unsqueeze(1)) ** 2,
        dim=1,
    ) / safe_occupied_mass
    # A plain `sqrt(12 * variance)` has an infinite derivative at zero variance,
    # which makes perfectly narrow single-bin occupancies produce NaN gradients.
    # The shifted square root preserves the zero-bandwidth fixed point while making
    # the local gradient finite and scaled to the dataset frequency resolution.
    occupied_bandwidth_hz = torch_module.sqrt(
        torch_module.clamp(12.0 * occupied_variance_hz2, min=0.0)
        + bandwidth_stabilizer_hz * bandwidth_stabilizer_hz,
    ) - bandwidth_stabilizer_hz
    occupied_bandwidth_hz = torch_module.clamp(
        occupied_bandwidth_hz,
        min=0.0,
    )
    occupied_bandwidth_hz = torch_module.where(
        occupied_mass > 1.0e-12,
        occupied_bandwidth_hz,
        torch_module.zeros_like(occupied_mass),
    )

    return TorchIllustrativeFeatureBatch(
        peak_frequency_hz=peak_frequency_hz,
        peak_power_db=peak_power_db,
        spectral_centroid_hz=centroid_hz,
        occupied_bandwidth_hz=occupied_bandwidth_hz,
    )


def _torch_feature_preservation_loss(
    reference_features: TorchIllustrativeFeatureBatch,
    reconstructed_features: TorchIllustrativeFeatureBatch,
    *,
    config: IllustrativeTaskConfig,
) -> Tensor:
    """Compute the batched differentiable feature-preservation surrogate."""
    torch_module = _require_torch()
    peak_term = _torch_huber(
        reference_features.peak_frequency_hz - reconstructed_features.peak_frequency_hz,
        delta=config.huber_delta,
    )
    peak_power_term = _torch_huber(
        reference_features.peak_power_db - reconstructed_features.peak_power_db,
        delta=config.huber_delta,
    )
    centroid_term = _torch_huber(
        reference_features.spectral_centroid_hz
        - reconstructed_features.spectral_centroid_hz,
        delta=config.huber_delta,
    )
    bandwidth_term = _torch_huber(
        reference_features.occupied_bandwidth_hz
        - reconstructed_features.occupied_bandwidth_hz,
        delta=config.huber_delta,
    )
    weighted = (
        config.peak_weight * peak_term
        + config.peak_power_weight * peak_power_term
        + config.centroid_weight * centroid_term
        + config.bandwidth_weight * bandwidth_term
    )
    return cast(Tensor, torch_module.mean(weighted))


def _torch_huber(
    values: Tensor,
    *,
    delta: float,
) -> Tensor:
    """Compute the elementwise Huber penalty used by the illustrative features."""
    torch_module = _require_torch()
    magnitudes = torch_module.abs(values)
    return cast(
        Tensor,
        torch_module.where(
            magnitudes <= delta,
            0.5 * magnitudes * magnitudes,
            delta * (magnitudes - 0.5 * delta),
        ),
    )


def _torch_frequency_grid_step_hz(
    frequency_grid_hz: Tensor,
) -> Tensor:
    """Return a stable positive frequency step used to regularize the bandwidth surrogate."""
    torch_module = _require_torch()
    if frequency_grid_hz.ndim != 1:
        raise CodecConfigurationError("frequency_grid_hz must be one-dimensional.")
    if frequency_grid_hz.shape[0] <= 1:
        return cast(
            Tensor,
            torch_module.tensor(
                1.0,
                dtype=frequency_grid_hz.dtype,
                device=frequency_grid_hz.device,
            ),
        )
    steps_hz = torch_module.abs(frequency_grid_hz[1:] - frequency_grid_hz[:-1])
    return cast(
        Tensor,
        torch_module.clamp(torch_module.mean(steps_hz), min=1.0),
    )


def _torch_moving_average(
    frames: Tensor,
    *,
    window_length: int,
) -> Tensor:
    """Apply the constant-spectrum-preserving moving-average smoother in torch."""
    torch_module = _require_torch()
    if window_length <= 0 or window_length % 2 == 0:
        raise CodecConfigurationError("window_length must be a positive odd integer.")
    if frames.ndim != 2:
        raise CodecConfigurationError("frames must have shape [batch, bin_count].")
    if window_length == 1:
        return frames

    radius = window_length // 2
    kernel = torch_module.full(
        (1, 1, window_length),
        1.0 / window_length,
        dtype=frames.dtype,
        device=frames.device,
    )
    padded = torch_module.nn.functional.pad(
        frames.unsqueeze(1),
        (radius, radius),
        mode="replicate",
    )
    smoothed = torch_module.nn.functional.conv1d(padded, kernel)
    return cast(Tensor, smoothed.squeeze(1))
