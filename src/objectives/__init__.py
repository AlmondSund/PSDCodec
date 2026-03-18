"""Distortion metrics and task-aware evaluation utilities for PSDCodec."""

from objectives.distortion import (
    DistortionBreakdown,
    IllustrativeFeatureSet,
    IllustrativeTaskConfig,
    estimate_reference_noise_floor,
    illustrative_task_loss,
    log_spectral_distortion,
)

__all__ = [
    "DistortionBreakdown",
    "IllustrativeFeatureSet",
    "IllustrativeTaskConfig",
    "estimate_reference_noise_floor",
    "illustrative_task_loss",
    "log_spectral_distortion",
]
