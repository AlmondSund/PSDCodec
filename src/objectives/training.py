"""Torch-compatible training losses for PSDCodec rate-distortion optimization."""

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
    logits_reference = (
        reference_frames - noise_floors - config.occupancy_margin
    ) / config.occupancy_temperature
    logits_reconstructed = (
        reconstructed_frames - noise_floors - config.occupancy_margin
    ) / config.occupancy_temperature
    reference_soft = torch_module.sigmoid(logits_reference)
    reconstructed_soft = torch_module.sigmoid(logits_reconstructed)
    clipped = torch_module.clamp(reconstructed_soft, min=1.0e-9, max=1.0 - 1.0e-9)
    positive = config.occupancy_positive_weight * reference_soft * torch_module.log(clipped)
    negative = (
        config.occupancy_negative_weight * (1.0 - reference_soft) * torch_module.log(1.0 - clipped)
    )
    return cast(Tensor, -torch_module.mean(positive + negative))


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
    """Compose the weighted training objective and return detached scalar metrics."""
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
    return total_loss, TrainingLossBreakdown(
        total_loss=float(total_loss.detach().cpu().item()),
        psd_loss=float(psd_loss.detach().cpu().item()),
        rate_bits=float(average_rate_bits.detach().cpu().item()),
        side_information_bits=average_side_information_bits,
        vq_loss=float(vq_loss.detach().cpu().item()),
        task_loss=float(total_task_loss.detach().cpu().item()),
    )
