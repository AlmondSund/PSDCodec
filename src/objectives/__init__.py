"""Distortion metrics and task-aware evaluation utilities for PSDCodec."""

from objectives.distortion import (
    DistortionBreakdown,
    IllustrativeFeatureSet,
    IllustrativeTaskBreakdown,
    IllustrativeTaskConfig,
    build_illustrative_task_breakdown,
    estimate_reference_noise_floor,
    extract_illustrative_features,
    hard_occupancy,
    illustrative_task_loss,
    log_spectral_distortion,
    soft_occupancy,
)
from objectives.training import (
    RateDistortionLossConfig,
    TorchIllustrativeFeatureBatch,
    TrainingLossBreakdown,
    compose_rate_distortion_loss,
    torch_illustrative_task_loss,
    torch_log_spectral_distortion,
    torch_occupancy_task_loss,
)

__all__ = [
    "DistortionBreakdown",
    "IllustrativeFeatureSet",
    "IllustrativeTaskBreakdown",
    "IllustrativeTaskConfig",
    "RateDistortionLossConfig",
    "TorchIllustrativeFeatureBatch",
    "TrainingLossBreakdown",
    "build_illustrative_task_breakdown",
    "compose_rate_distortion_loss",
    "estimate_reference_noise_floor",
    "extract_illustrative_features",
    "hard_occupancy",
    "illustrative_task_loss",
    "log_spectral_distortion",
    "soft_occupancy",
    "torch_illustrative_task_loss",
    "torch_log_spectral_distortion",
    "torch_occupancy_task_loss",
]
