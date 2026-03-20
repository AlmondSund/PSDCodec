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
    peak_power_weight: float = 1.0  # α_pk,pwr
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
        if self.peak_weight < 0.0:
            raise CodecConfigurationError("peak_weight must be non-negative.")
        if self.peak_power_weight < 0.0:
            raise CodecConfigurationError("peak_power_weight must be non-negative.")
        if self.centroid_weight < 0.0:
            raise CodecConfigurationError("centroid_weight must be non-negative.")
        if self.bandwidth_weight < 0.0:
            raise CodecConfigurationError("bandwidth_weight must be non-negative.")
        if self.occupancy_weight < 0.0:
            raise CodecConfigurationError("occupancy_weight must be non-negative.")
        if self.feature_weight < 0.0:
            raise CodecConfigurationError("feature_weight must be non-negative.")


@dataclass(frozen=True)
class IllustrativeFeatureSet:
    """Illustrative sensing features derived from one PSD frame."""

    peak_frequency_hz: float  # Dominant peak location after smoothing
    peak_power_db: float  # Dominant peak amplitude expressed in dB
    spectral_centroid_hz: float  # Energy-weighted spectral centroid
    occupied_bandwidth_hz: float  # Width of the dominant occupied component


@dataclass(frozen=True)
class DistortionBreakdown:
    """Distortion decomposition used for preprocessing-vs-codec ablations."""

    psd_distortion: float  # D_psd(s_t, ŝ_t)
    preprocessing_distortion: float  # Δ_pre
    codec_distortion: float  # Δ_codec
    task_distortion: float | None = None  # Optional D_task


@dataclass(frozen=True)
class IllustrativeTaskBreakdown:
    """Full breakdown of the manuscript's illustrative sensing task example.

    This object exposes the exact occupancy and feature terms described in the paper's
    illustrative sensing instantiation so notebooks and evaluations can inspect what
    the optional task regularizer is rewarding or penalizing.
    """

    occupancy_loss: float  # D_occ
    feature_loss: float  # D_feat
    total_loss: float  # D_task,ex = β_occ D_occ + β_feat D_feat
    reference_soft_occupancy: FloatArray  # p_t
    reconstructed_soft_occupancy: FloatArray  # p̂_t
    reference_hard_occupancy: np.ndarray  # o_t
    reconstructed_hard_occupancy: np.ndarray  # ô_t
    reference_features: IllustrativeFeatureSet  # f_ex from the reference frame
    reconstructed_features: IllustrativeFeatureSet  # f̂_ex from the reconstructed frame


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
    return build_illustrative_task_breakdown(
        reference_frame,
        reconstructed_frame,
        noise_floor=noise_floor,
        frequency_grid_hz=frequency_grid_hz,
        config=config,
    ).total_loss


def build_illustrative_task_breakdown(
    reference_frame: FloatArray,  # Ground-truth PSD frame s_t
    reconstructed_frame: FloatArray,  # Reconstructed PSD frame ŝ_t
    *,
    noise_floor: FloatArray,  # Reference noise floor \bar{n}_t
    frequency_grid_hz: FloatArray,  # Frequency grid ω_n expressed in Hz
    config: IllustrativeTaskConfig,
) -> IllustrativeTaskBreakdown:
    """Return the full illustrative sensing-task breakdown from the manuscript.

    Purpose:
        Expose the exact occupancy and feature-preservation quantities used by the
        paper's example task so demos can inspect both the scalar loss components and
        the intermediate occupancy/feature representations.
    """
    reference, reconstructed, baseline, frequency_grid = _validate_task_inputs(
        reference_frame,
        reconstructed_frame,
        noise_floor=noise_floor,
        frequency_grid_hz=frequency_grid_hz,
    )
    reference_soft = soft_occupancy(
        reference,
        noise_floor=baseline,
        config=config,
    )
    reconstructed_soft = soft_occupancy(
        reconstructed,
        noise_floor=baseline,
        config=config,
    )
    occupancy_term = _occupancy_consistency(reference_soft, reconstructed_soft, config)

    reference_hard = hard_occupancy(
        reference,
        noise_floor=baseline,
        config=config,
    )
    reconstructed_hard = hard_occupancy(
        reconstructed,
        noise_floor=baseline,
        config=config,
    )
    reference_features = extract_illustrative_features(
        reference,
        frequency_grid_hz=frequency_grid,
        occupancy_mask=reference_hard,
        smoothing_window_bins=config.smoothing_window_bins,
    )
    reconstructed_features = extract_illustrative_features(
        reconstructed,
        frequency_grid_hz=frequency_grid,
        occupancy_mask=reconstructed_hard,
        smoothing_window_bins=config.smoothing_window_bins,
    )
    feature_term = _feature_preservation_loss(
        reference_features,
        reconstructed_features,
        config=config,
    )
    total_loss = config.occupancy_weight * occupancy_term + config.feature_weight * feature_term
    return IllustrativeTaskBreakdown(
        occupancy_loss=occupancy_term,
        feature_loss=feature_term,
        total_loss=total_loss,
        reference_soft_occupancy=reference_soft,
        reconstructed_soft_occupancy=reconstructed_soft,
        reference_hard_occupancy=reference_hard,
        reconstructed_hard_occupancy=reconstructed_hard,
        reference_features=reference_features,
        reconstructed_features=reconstructed_features,
    )


def soft_occupancy(
    frame: FloatArray,  # PSD frame to threshold softly
    *,
    noise_floor: FloatArray,  # Reference baseline
    config: IllustrativeTaskConfig,
) -> FloatArray:
    """Compute the manuscript soft occupancy proxy p_{t,n} for one PSD frame."""
    candidate_frame = as_1d_float_array(frame, name="frame")
    baseline = as_1d_float_array(noise_floor, name="noise_floor")
    if candidate_frame.shape != baseline.shape:
        raise CodecConfigurationError("frame and noise_floor must have the same shape.")
    return _soft_occupancy(candidate_frame, baseline, config)


def hard_occupancy(
    frame: FloatArray,  # PSD frame to threshold after soft occupancy
    *,
    noise_floor: FloatArray,  # Reference baseline
    config: IllustrativeTaskConfig,
) -> np.ndarray:
    """Compute the manuscript hard occupancy mask o_{t,n} for one PSD frame."""
    return soft_occupancy(
        frame,
        noise_floor=noise_floor,
        config=config,
    ) >= 0.5


def extract_illustrative_features(
    frame: FloatArray,  # PSD frame from which features are extracted
    *,
    frequency_grid_hz: FloatArray,  # Frequency support ω_n
    occupancy_mask: np.ndarray,  # Hard occupancy mask o_{t,n}
    smoothing_window_bins: int,
) -> IllustrativeFeatureSet:
    """Extract the manuscript illustrative features from one PSD frame.

    Args:
        frame: PSD frame in linear power.
        frequency_grid_hz: Uniform frequency support aligned with `frame`.
        occupancy_mask: Boolean occupancy mask defined from the same frequency grid.
        smoothing_window_bins: Odd moving-average window used only for peak extraction.
    """
    candidate_frame = as_1d_float_array(frame, name="frame")
    frequency_grid = as_1d_float_array(
        frequency_grid_hz,
        name="frequency_grid_hz",
        allow_negative=True,
    )
    mask = np.asarray(occupancy_mask, dtype=bool)
    if candidate_frame.shape != frequency_grid.shape or candidate_frame.shape != mask.shape:
        raise CodecConfigurationError(
            "frame, frequency_grid_hz, and occupancy_mask must have the same shape."
        )
    if smoothing_window_bins <= 0 or smoothing_window_bins % 2 == 0:
        raise CodecConfigurationError(
            "smoothing_window_bins must be a positive odd integer."
        )
    return _extract_illustrative_features(
        candidate_frame,
        frequency_grid,
        mask,
        smoothing_window_bins=smoothing_window_bins,
    )


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

    # Use a numerically stable logistic evaluation so high-SNR campaign frames do not
    # emit overflow warnings while still saturating cleanly toward 0 or 1.
    probabilities = np.empty_like(logits, dtype=np.float64)
    positive_mask = logits >= 0.0
    probabilities[positive_mask] = 1.0 / (1.0 + np.exp(-logits[positive_mask]))
    negative_logits = np.exp(logits[~positive_mask])
    probabilities[~positive_mask] = negative_logits / (1.0 + negative_logits)
    return probabilities


def _validate_task_inputs(
    reference_frame: FloatArray,
    reconstructed_frame: FloatArray,
    *,
    noise_floor: FloatArray,
    frequency_grid_hz: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Validate the aligned arrays needed by the illustrative task example."""
    reference = as_1d_float_array(reference_frame, name="reference_frame")
    reconstructed = as_1d_float_array(reconstructed_frame, name="reconstructed_frame")
    baseline = as_1d_float_array(noise_floor, name="noise_floor")
    frequency_grid = as_1d_float_array(
        frequency_grid_hz,
        name="frequency_grid_hz",
        allow_negative=True,
    )
    if not (reference.shape == reconstructed.shape == baseline.shape == frequency_grid.shape):
        raise CodecConfigurationError(
            "reference_frame, reconstructed_frame, noise_floor, and frequency_grid_hz must align."
        )
    return reference, reconstructed, baseline, frequency_grid


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


def _peak_power_db_from_frame(
    frame: FloatArray,  # PSD frame from which the dominant raw peak is read
) -> float:
    """Return the dominant raw peak amplitude in dB for one PSD frame."""
    peak_power_linear = float(np.max(frame))
    return float(10.0 * np.log10(max(peak_power_linear, 1.0e-12)))


def _extract_illustrative_features(
    frame: FloatArray,  # PSD frame from which features are extracted
    frequency_grid_hz: FloatArray,  # Frequency support ω_n
    occupancy_mask: np.ndarray,  # Hard occupancy mask o_{t,n}
    *,
    smoothing_window_bins: int,
) -> IllustrativeFeatureSet:
    """Extract the illustrative peak, peak-power, centroid, and bandwidth features."""
    smoothed = _moving_average(frame, window_length=smoothing_window_bins)
    peak_index = int(np.argmax(smoothed))
    total_power = float(np.sum(frame))
    centroid = float(np.sum(frequency_grid_hz * frame) / total_power) if total_power > 0.0 else 0.0

    dominant_component_power = 0.0
    dominant_bandwidth_hz = 0.0
    for component in _connected_components(occupancy_mask):
        component_power = float(np.sum(frame[component]))
        if component_power <= 0.0:
            continue
        current_bandwidth = float(
            frequency_grid_hz[component.stop - 1] - frequency_grid_hz[component.start]
        )
        # The manuscript selects the occupied connected component with the largest
        # integrated spectral mass, then reports that component's bandwidth.
        if component_power >= dominant_component_power:
            dominant_component_power = component_power
            dominant_bandwidth_hz = current_bandwidth

    return IllustrativeFeatureSet(
        peak_frequency_hz=float(frequency_grid_hz[peak_index]),
        peak_power_db=_peak_power_db_from_frame(frame),
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
    peak_power_term = _huber(
        reference_features.peak_power_db - reconstructed_features.peak_power_db,
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
        + config.peak_power_weight * peak_power_term
        + config.centroid_weight * centroid_term
        + config.bandwidth_weight * bandwidth_term
    )


def _moving_average(
    frame: FloatArray,  # PSD frame to smooth
    *,
    window_length: int,
) -> FloatArray:
    """Smooth a frame with a constant-spectrum-preserving moving-average filter."""
    if window_length <= 0 or window_length % 2 == 0:
        raise CodecConfigurationError("window_length must be a positive odd integer.")
    if window_length == 1:
        return frame.copy()

    # Edge replication preserves both non-negativity and constant spectra, which is
    # the admissibility requirement stated in the manuscript for the feature smoother.
    radius = window_length // 2
    padded = np.pad(frame, pad_width=radius, mode="edge")
    kernel = np.full(window_length, 1.0 / window_length, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


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
