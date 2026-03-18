"""Distortion metrics and task-aware evaluation utilities for PSDCodec."""

from objectives.distortion import (
    DistortionBreakdown,
    IllustrativeFeatureSet,
    IllustrativeTaskConfig,
    estimate_reference_noise_floor,
    illustrative_task_loss,
    log_spectral_distortion,
)
from objectives.training import (
    RateDistortionLossConfig,
    TrainingLossBreakdown,
    compose_rate_distortion_loss,
    torch_log_spectral_distortion,
    torch_occupancy_task_loss,
)

__all__ = [
    "DistortionBreakdown",
    "IllustrativeFeatureSet",
    "IllustrativeTaskConfig",
    "RateDistortionLossConfig",
    "TrainingLossBreakdown",
    "compose_rate_distortion_loss",
    "estimate_reference_noise_floor",
    "illustrative_task_loss",
    "log_spectral_distortion",
    "torch_log_spectral_distortion",
    "torch_occupancy_task_loss",
]
