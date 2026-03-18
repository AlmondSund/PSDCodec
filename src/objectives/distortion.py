"""PSD-domain distortion metrics and optional sensing-oriented task losses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from codec.exceptions import CodecConfigurationError
from utils import FloatArray, as_1d_float_array


@dataclass(frozen=True)
class IllustrativeTaskConfig:
    """Weights and hyperparameters for the illustrative sensing task regularizer."""

    occupancy_margin: float  # Margin γ applied above the reference noise floor
    occupancy_temperature: float = 0.25  # Soft-threshold temperature τ_occ
    occupancy_positive_weight: float = 2.0  # Asymmetric missed-detection penalty w_1
    occupancy_negative_weight: float = 1.0  # False-alarm penalty w_0
    smoothing_window_bins: int = 5  # Odd moving-average window used only for features
    huber_delta: float = 1.0  # Huber threshold δ
    peak_weight: float = 1.0  # α_pk
    centroid_weight: float = 1.0  # α_cent
    bandwidth_weight: float = 1.0  # α_bw
    occupancy_weight: float = 1.0  # β_occ
    feature_weight: float = 1.0  # β_feat

    def __post_init__(self) -> None:
        """Validate the illustrative task configuration."""
        if self.occupancy_temperature <= 0.0:
            raise CodecConfigurationError("occupancy_temperature must be strictly positive.")
        if self.smoothing_window_bins <= 0 or self.smoothing_window_bins % 2 == 0:
            raise CodecConfigurationError("smoothing_window_bins must be a positive odd integer.")
        if self.huber_delta <= 0.0:
            raise CodecConfigurationError("huber_delta must be strictly positive.")


@dataclass(frozen=True)
class IllustrativeFeatureSet:
    """Illustrative sensing features derived from one PSD frame."""

    peak_frequency_hz: float  # Dominant peak location after smoothing
    spectral_centroid_hz: float  # Energy-weighted spectral centroid
    occupied_bandwidth_hz: float  # Width of the dominant occupied component


@dataclass(frozen=True)
class DistortionBreakdown:
    """Distortion decomposition used for preprocessing-vs-codec ablations."""

    psd_distortion: float  # D_psd(s_t, ŝ_t)
    preprocessing_distortion: float  # Δ_pre
    codec_distortion: float  # Δ_codec
    task_distortion: float | None = None  # Optional D_task


def log_spectral_distortion(
    reference_frame: FloatArray,  # Ground-truth PSD frame s_t
    reconstructed_frame: FloatArray,  # Reconstructed PSD frame ŝ_t
    *,
    dynamic_range_offset: float,  # Positive κ inside the log distortion
) -> float:
    """Compute the core PSD distortion term from the manuscript."""
    reference = as_1d_float_array(reference_frame, name="reference_frame")
    reconstructed = as_1d_float_array(reconstructed_frame, name="reconstructed_frame")
    if reference.shape != reconstructed.shape:
        raise CodecConfigurationError(
            "reference_frame and reconstructed_frame must have the same shape."
        )
    if dynamic_range_offset <= 0.0:
        raise CodecConfigurationError("dynamic_range_offset must be strictly positive.")

    difference = np.log(reference + dynamic_range_offset) - np.log(
        reconstructed + dynamic_range_offset
    )
    return float(np.mean(difference * difference))


def estimate_reference_noise_floor(
    frames: np.ndarray,  # Window of uncompressed PSD frames with shape [W, N]
    *,
    percentile: float = 10.0,  # Lower percentile used as a robust baseline
) -> FloatArray:
    """Estimate the reference noise floor from uncompressed historical frames."""
    frame_matrix = np.asarray(frames, dtype=np.float64)
    if frame_matrix.ndim != 2:
        raise CodecConfigurationError("frames must have shape [window_length, bin_count].")
    if frame_matrix.shape[0] < 1:
        raise CodecConfigurationError("frames must contain at least one PSD frame.")
    if np.any(frame_matrix < 0.0) or not np.all(np.isfinite(frame_matrix)):
        raise CodecConfigurationError("frames must contain finite non-negative PSD values.")
    if not (0.0 <= percentile <= 100.0):
        raise CodecConfigurationError("percentile must lie in [0, 100].")
    return np.percentile(frame_matrix, percentile, axis=0).astype(np.float64)


def illustrative_task_loss(
    reference_frame: FloatArray,  # Ground-truth PSD frame s_t
    reconstructed_frame: FloatArray,  # Reconstructed PSD frame ŝ_t
    *,
    noise_floor: FloatArray,  # Reference noise floor \bar{n}_t
    frequency_grid_hz: FloatArray,  # Frequency grid ω_n expressed in Hz
    config: IllustrativeTaskConfig,
) -> float:
    """Compute the illustrative occupancy-plus-feature task regularizer."""
    reference = as_1d_float_array(reference_frame, name="reference_frame")
    reconstructed = as_1d_float_array(reconstructed_frame, name="reconstructed_frame")
    baseline = as_1d_float_array(noise_floor, name="noise_floor")
    frequency_grid = as_1d_float_array(
        frequency_grid_hz, name="frequency_grid_hz", allow_negative=True
    )
    if not (reference.shape == reconstructed.shape == baseline.shape == frequency_grid.shape):
        raise CodecConfigurationError(
            "reference_frame, reconstructed_frame, noise_floor, and frequency_grid_hz must align."
        )

    reference_soft = _soft_occupancy(reference, baseline, config)
    reconstructed_soft = _soft_occupancy(reconstructed, baseline, config)
    occupancy_term = _occupancy_consistency(reference_soft, reconstructed_soft, config)

    reference_hard = reference_soft >= 0.5
    reconstructed_hard = reconstructed_soft >= 0.5
    reference_features = _extract_illustrative_features(
        reference,
        frequency_grid,
        reference_hard,
        smoothing_window_bins=config.smoothing_window_bins,
    )
    reconstructed_features = _extract_illustrative_features(
        reconstructed,
        frequency_grid,
        reconstructed_hard,
        smoothing_window_bins=config.smoothing_window_bins,
    )
    feature_term = _feature_preservation_loss(
        reference_features,
        reconstructed_features,
        config=config,
    )
    return config.occupancy_weight * occupancy_term + config.feature_weight * feature_term


def build_distortion_breakdown(
    reference_frame: FloatArray,  # Ground-truth PSD frame s_t
    preprocessing_only_frame: FloatArray,  # Preprocessing-only reconstruction \bar{s}_t
    reconstructed_frame: FloatArray,  # Full codec reconstruction ŝ_t
    *,
    dynamic_range_offset: float,  # Positive κ inside the log distortion
    task_distortion: float | None = None,  # Optional D_task value
) -> DistortionBreakdown:
    """Build the preprocessing-vs-codec ablation breakdown from three aligned frames."""
    psd_distortion = log_spectral_distortion(
        reference_frame,
        reconstructed_frame,
        dynamic_range_offset=dynamic_range_offset,
    )
    preprocessing_distortion = log_spectral_distortion(
        reference_frame,
        preprocessing_only_frame,
        dynamic_range_offset=dynamic_range_offset,
    )
    codec_distortion = log_spectral_distortion(
        preprocessing_only_frame,
        reconstructed_frame,
        dynamic_range_offset=dynamic_range_offset,
    )
    return DistortionBreakdown(
        psd_distortion=psd_distortion,
        preprocessing_distortion=preprocessing_distortion,
        codec_distortion=codec_distortion,
        task_distortion=task_distortion,
    )


def _soft_occupancy(
    frame: FloatArray,  # PSD frame to threshold softly
    noise_floor: FloatArray,  # Reference baseline
    config: IllustrativeTaskConfig,
) -> FloatArray:
    """Compute the differentiable occupancy proxy p_{t,n}."""
    logits = (frame - noise_floor - config.occupancy_margin) / config.occupancy_temperature
    return 1.0 / (1.0 + np.exp(-logits))


def _occupancy_consistency(
    reference_soft: FloatArray,  # Soft occupancies p_{t,n}
    reconstructed_soft: FloatArray,  # Soft occupancies p̂_{t,n}
    config: IllustrativeTaskConfig,
) -> float:
    """Compute the weighted occupancy cross-entropy term D_occ."""
    clipped = np.clip(reconstructed_soft, 1.0e-9, 1.0 - 1.0e-9)
    positive = config.occupancy_positive_weight * reference_soft * np.log(clipped)
    negative = config.occupancy_negative_weight * (1.0 - reference_soft) * np.log(1.0 - clipped)
    return float(-np.mean(positive + negative))


def _extract_illustrative_features(
    frame: FloatArray,  # PSD frame from which features are extracted
    frequency_grid_hz: FloatArray,  # Frequency support ω_n
    occupancy_mask: np.ndarray,  # Hard occupancy mask o_{t,n}
    *,
    smoothing_window_bins: int,
) -> IllustrativeFeatureSet:
    """Extract the illustrative peak, centroid, and occupied-bandwidth features."""
    smoothed = _moving_average(frame, window_length=smoothing_window_bins)
    peak_index = int(np.argmax(smoothed))
    total_power = float(np.sum(frame))
    centroid = float(np.sum(frequency_grid_hz * frame) / total_power) if total_power > 0.0 else 0.0

    dominant_bandwidth_hz = 0.0
    for component in _connected_components(occupancy_mask):
        component_power = float(np.sum(frame[component]))
        if component_power <= 0.0:
            continue
        current_bandwidth = float(
            frequency_grid_hz[component.stop - 1] - frequency_grid_hz[component.start]
        )
        if current_bandwidth >= dominant_bandwidth_hz:
            dominant_bandwidth_hz = current_bandwidth

    return IllustrativeFeatureSet(
        peak_frequency_hz=float(frequency_grid_hz[peak_index]),
        spectral_centroid_hz=centroid,
        occupied_bandwidth_hz=dominant_bandwidth_hz,
    )


def _feature_preservation_loss(
    reference_features: IllustrativeFeatureSet,  # Features from the original frame
    reconstructed_features: IllustrativeFeatureSet,  # Features from the reconstructed frame
    *,
    config: IllustrativeTaskConfig,
) -> float:
    """Compute the illustrative Huber feature-preservation term D_feat."""
    peak_term = _huber(
        reference_features.peak_frequency_hz - reconstructed_features.peak_frequency_hz,
        delta=config.huber_delta,
    )
    centroid_term = _huber(
        reference_features.spectral_centroid_hz - reconstructed_features.spectral_centroid_hz,
        delta=config.huber_delta,
    )
    bandwidth_term = _huber(
        reference_features.occupied_bandwidth_hz - reconstructed_features.occupied_bandwidth_hz,
        delta=config.huber_delta,
    )
    return (
        config.peak_weight * peak_term
        + config.centroid_weight * centroid_term
        + config.bandwidth_weight * bandwidth_term
    )


def _moving_average(
    frame: FloatArray,  # PSD frame to smooth
    *,
    window_length: int,
) -> FloatArray:
    """Smooth a frame with a normalized moving-average filter."""
    kernel = np.full(window_length, 1.0 / window_length, dtype=np.float64)
    return np.convolve(frame, kernel, mode="same")


def _connected_components(mask: np.ndarray) -> tuple[slice, ...]:
    """Return contiguous `True` runs in a one-dimensional boolean mask."""
    components: list[slice] = []
    start: int | None = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            components.append(slice(start, index))
            start = None
    if start is not None:
        components.append(slice(start, len(mask)))
    return tuple(components)


def _huber(
    value: float,  # Scalar feature mismatch
    *,
    delta: float,  # Huber threshold δ
) -> float:
    """Compute the scalar Huber penalty ρ_δ."""
    magnitude = abs(value)
    if magnitude <= delta:
        return 0.5 * magnitude * magnitude
    return delta * (magnitude - 0.5 * delta)
