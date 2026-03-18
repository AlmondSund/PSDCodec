"""Raw campaign ingestion and harmonization utilities for PSD acquisition datasets."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, cast

import numpy as np

from codec.exceptions import CodecConfigurationError
from objectives.distortion import estimate_reference_noise_floor

csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class CampaignDatasetBundle:
    """In-memory representation of raw campaign PSD acquisitions.

    The raw campaign CSV files store PSD values as serialized lists inside the `pxx`
    column. This bundle normalizes those heterogeneous files into batch-compatible
    NumPy arrays that can be saved to `data/processed/` or passed directly to the
    training dataset layer.
    """

    frames: np.ndarray  # Harmonized PSD frames with shape [num_frames, N]
    frequency_grid_hz: np.ndarray  # Shared frequency grid with shape [N]
    noise_floors: np.ndarray | None = None  # Sequence-local reference noise floors
    campaign_labels: np.ndarray | None = None  # Campaign label per frame
    campaign_ids: np.ndarray | None = None  # Campaign identifier per frame
    node_labels: np.ndarray | None = None  # Node file stem per frame
    sequence_ids: np.ndarray | None = None  # Stable `campaign/node` identifier per frame
    timestamps_ms: np.ndarray | None = None  # Acquisition timestamp per frame [ms since epoch]


@dataclass(frozen=True)
class _CampaignMetadata:
    """Campaign-level metadata parsed from one `metadata.csv` file."""

    campaign_label: str
    campaign_id: int


@dataclass(frozen=True)
class _NodeMeasurement:
    """One PSD acquisition row parsed from a node CSV file."""

    frame: np.ndarray  # PSD values in the selected value domain
    start_freq_hz: float  # Lower frequency support bound [Hz]
    end_freq_hz: float  # Upper frequency support bound [Hz]
    timestamp_ms: int  # Acquisition timestamp [ms since epoch]


def load_campaign_dataset_bundle(
    campaign_root: str | Path,
    *,
    include_campaign_globs: list[str] | tuple[str, ...] | None = None,
    exclude_campaign_globs: list[str] | tuple[str, ...] | None = None,
    include_node_globs: list[str] | tuple[str, ...] | None = None,
    target_bin_count: int | None = None,
    value_scale: str = "db_to_power",
    max_frames: int | None = None,
    noise_floor_window: int | None = None,
    noise_floor_percentile: float = 10.0,
) -> CampaignDatasetBundle:
    """Load raw campaign PSD acquisitions into a harmonized in-memory bundle.

    Args:
        campaign_root: Root directory containing campaign subdirectories.
        include_campaign_globs: Campaign-directory glob filters. Defaults to `["*"]`.
        exclude_campaign_globs: Campaign-directory glob filters excluded after inclusion.
        include_node_globs: Node-file glob filters. Defaults to `["Node*.csv"]`.
        target_bin_count: Optional common output frequency-grid length. When omitted,
            all selected frames must already share a common bin count.
        value_scale: Raw PSD value conversion. `"db_to_power"` converts the campaign
            `pxx` lists from dB-like values into non-negative linear power via `10^(x/10)`.
            `"identity"` keeps the stored values unchanged.
        max_frames: Optional upper bound on the number of returned frames after
            deterministic campaign/node/timestamp ordering.
        noise_floor_window: Optional per-sequence history window used to estimate
            reference noise floors. The history is computed inside each campaign/node
            sequence and never across unrelated acquisitions.
        noise_floor_percentile: Percentile used by the sequence-local noise-floor estimator.

    Returns:
        A batch-compatible campaign bundle with harmonized PSD frames and metadata.

    Raises:
        CodecConfigurationError: If the selected campaign files are inconsistent or if
            the requested harmonization cannot be satisfied.
    """
    root_path = Path(campaign_root)
    if not root_path.exists():
        raise CodecConfigurationError(f"Campaign root does not exist: {root_path}.")
    if not root_path.is_dir():
        raise CodecConfigurationError(f"Campaign root must be a directory: {root_path}.")
    if target_bin_count is not None and target_bin_count <= 0:
        raise CodecConfigurationError("target_bin_count must be strictly positive when set.")
    if max_frames is not None and max_frames <= 0:
        raise CodecConfigurationError("max_frames must be strictly positive when set.")
    if noise_floor_window is not None and noise_floor_window <= 0:
        raise CodecConfigurationError("noise_floor_window must be strictly positive when set.")
    if not (0.0 <= noise_floor_percentile <= 100.0):
        raise CodecConfigurationError("noise_floor_percentile must lie in [0, 100].")

    include_campaigns = list(include_campaign_globs or ["*"])
    exclude_campaigns = list(exclude_campaign_globs or [])
    include_nodes = list(include_node_globs or ["Node*.csv"])

    frames: list[np.ndarray] = []
    campaign_labels: list[str] = []
    campaign_ids: list[int] = []
    node_labels: list[str] = []
    sequence_ids: list[str] = []
    timestamps_ms: list[int] = []
    noise_floors: list[np.ndarray] | None = [] if noise_floor_window is not None else None

    shared_frequency_bounds_hz: tuple[float, float] | None = None
    shared_frequency_grid_hz: np.ndarray | None = None

    for campaign_dir in _select_campaign_directories(
        root_path,
        include_campaigns=include_campaigns,
        exclude_campaigns=exclude_campaigns,
    ):
        metadata = _load_campaign_metadata(campaign_dir / "metadata.csv")
        node_paths = _select_node_files(campaign_dir, include_node_globs=include_nodes)
        for node_path in node_paths:
            node_measurements = _load_node_measurements(node_path, value_scale=value_scale)
            node_measurements.sort(key=lambda measurement: measurement.timestamp_ms)
            sequence_frames: list[np.ndarray] = []
            sequence_timestamps_ms: list[int] = []

            # Harmonize every frame from this sequence onto the shared frequency grid.
            for measurement in node_measurements:
                measurement_bounds_hz = (
                    measurement.start_freq_hz,
                    measurement.end_freq_hz,
                )
                if shared_frequency_bounds_hz is None:
                    shared_frequency_bounds_hz = measurement_bounds_hz
                elif measurement_bounds_hz != shared_frequency_bounds_hz:
                    raise CodecConfigurationError(
                        "Selected raw campaigns do not share a common frequency support."
                    )

                source_grid_hz = _build_uniform_frequency_grid(
                    measurement.start_freq_hz,
                    measurement.end_freq_hz,
                    measurement.frame.size,
                )
                if shared_frequency_grid_hz is None:
                    resolved_bin_count = (
                        measurement.frame.size if target_bin_count is None else target_bin_count
                    )
                    shared_frequency_grid_hz = _build_uniform_frequency_grid(
                        measurement.start_freq_hz,
                        measurement.end_freq_hz,
                        resolved_bin_count,
                    )
                if (
                    target_bin_count is None
                    and measurement.frame.size != shared_frequency_grid_hz.size
                ):
                    raise CodecConfigurationError(
                        "Selected raw campaigns have mixed PSD lengths. Set target_bin_count "
                        "to harmonize them onto one grid."
                    )

                sequence_frames.append(
                    _resample_frame_to_grid(
                        measurement.frame,
                        source_grid_hz=source_grid_hz,
                        target_grid_hz=shared_frequency_grid_hz,
                    )
                )
                sequence_timestamps_ms.append(measurement.timestamp_ms)

            if not sequence_frames:
                continue

            sequence_noise_floors = (
                None
                if noise_floor_window is None
                else _estimate_sequence_noise_floors(
                    np.stack(sequence_frames, axis=0),
                    window_length=noise_floor_window,
                    percentile=noise_floor_percentile,
                )
            )
            sequence_id = f"{campaign_dir.name}/{node_path.stem}"

            # Append the sequence in timestamp order so temporal context remains well defined.
            for frame_index, frame in enumerate(sequence_frames):
                frames.append(frame)
                campaign_labels.append(metadata.campaign_label)
                campaign_ids.append(metadata.campaign_id)
                node_labels.append(node_path.stem)
                sequence_ids.append(sequence_id)
                timestamps_ms.append(sequence_timestamps_ms[frame_index])
                if noise_floors is not None and sequence_noise_floors is not None:
                    noise_floors.append(sequence_noise_floors[frame_index])

                if max_frames is not None and len(frames) >= max_frames:
                    return _build_campaign_bundle(
                        frames=frames,
                        frequency_grid_hz=shared_frequency_grid_hz,
                        noise_floors=noise_floors,
                        campaign_labels=campaign_labels,
                        campaign_ids=campaign_ids,
                        node_labels=node_labels,
                        sequence_ids=sequence_ids,
                        timestamps_ms=timestamps_ms,
                    )

    return _build_campaign_bundle(
        frames=frames,
        frequency_grid_hz=shared_frequency_grid_hz,
        noise_floors=noise_floors,
        campaign_labels=campaign_labels,
        campaign_ids=campaign_ids,
        node_labels=node_labels,
        sequence_ids=sequence_ids,
        timestamps_ms=timestamps_ms,
    )


def save_campaign_dataset_bundle(
    bundle: CampaignDatasetBundle,
    output_path: str | Path,
) -> Path:
    """Persist a campaign bundle to a compressed `.npz` archive."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "frames": bundle.frames,
        "frequency_grid_hz": bundle.frequency_grid_hz,
    }
    if bundle.noise_floors is not None:
        payload["noise_floors"] = bundle.noise_floors
    if bundle.campaign_labels is not None:
        payload["campaign_labels"] = bundle.campaign_labels
    if bundle.campaign_ids is not None:
        payload["campaign_ids"] = bundle.campaign_ids
    if bundle.node_labels is not None:
        payload["node_labels"] = bundle.node_labels
    if bundle.sequence_ids is not None:
        payload["sequence_ids"] = bundle.sequence_ids
    if bundle.timestamps_ms is not None:
        payload["timestamps_ms"] = bundle.timestamps_ms
    np.savez_compressed(target_path, **cast(Any, payload))
    return target_path


def _select_campaign_directories(
    root_path: Path,
    *,
    include_campaigns: list[str],
    exclude_campaigns: list[str],
) -> list[Path]:
    """Return campaign directories selected by the provided glob filters."""
    selected = [
        path
        for path in sorted(root_path.iterdir())
        if path.is_dir()
        and _matches_any(path.name, include_campaigns)
        and not _matches_any(path.name, exclude_campaigns)
    ]
    if not selected:
        raise CodecConfigurationError("No campaign directories matched the configured filters.")
    return selected


def _select_node_files(
    campaign_dir: Path,
    *,
    include_node_globs: list[str],
) -> list[Path]:
    """Return node CSV files from one campaign directory."""
    node_paths = [
        path
        for path in sorted(campaign_dir.glob("*.csv"))
        if path.name != "metadata.csv" and _matches_any(path.name, include_node_globs)
    ]
    if not node_paths:
        raise CodecConfigurationError(
            f"No node CSV files matched the configured filters in {campaign_dir}."
        )
    return node_paths


def _load_campaign_metadata(metadata_path: Path) -> _CampaignMetadata:
    """Parse the small campaign-level metadata file."""
    if not metadata_path.exists():
        raise CodecConfigurationError(f"Missing campaign metadata file: {metadata_path}.")
    with metadata_path.open("r", encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    return _CampaignMetadata(
        campaign_label=str(row["campaign_label"]),
        campaign_id=int(row["campaign_id"]),
    )


def _load_node_measurements(
    node_path: Path,
    *,
    value_scale: str,
) -> list[_NodeMeasurement]:
    """Parse one node CSV file into typed PSD measurements."""
    measurements: list[_NodeMeasurement] = []
    with node_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            raw_values = np.asarray(json.loads(row["pxx"]), dtype=np.float64)
            if raw_values.ndim != 1 or raw_values.size == 0:
                raise CodecConfigurationError(
                    f"Node CSV contains an invalid PSD vector: {node_path}."
                )
            frame = _convert_raw_psd_values(raw_values, value_scale=value_scale)
            measurements.append(
                _NodeMeasurement(
                    frame=frame,
                    start_freq_hz=float(row["start_freq_hz"]),
                    end_freq_hz=float(row["end_freq_hz"]),
                    timestamp_ms=int(row["timestamp"]),
                )
            )
    if not measurements:
        raise CodecConfigurationError(f"Node CSV contains no PSD measurements: {node_path}.")
    return measurements


def _convert_raw_psd_values(
    raw_values: np.ndarray,
    *,
    value_scale: str,
) -> np.ndarray:
    """Convert raw campaign PSD values into the domain expected by the codec."""
    if not np.all(np.isfinite(raw_values)):
        raise CodecConfigurationError("Raw campaign PSD values must be finite.")
    if value_scale == "db_to_power":
        return np.power(10.0, raw_values / 10.0, dtype=np.float64)
    if value_scale == "identity":
        return raw_values.astype(np.float64, copy=False)
    raise CodecConfigurationError(
        f"Unsupported raw campaign value_scale '{value_scale}'. "
        "Expected 'db_to_power' or 'identity'."
    )


def _build_uniform_frequency_grid(
    start_freq_hz: float,
    end_freq_hz: float,
    bin_count: int,
) -> np.ndarray:
    """Create a uniform frequency grid consistent with the campaign support bounds."""
    if bin_count <= 0:
        raise CodecConfigurationError("bin_count must be strictly positive.")
    if end_freq_hz <= start_freq_hz:
        raise CodecConfigurationError("end_freq_hz must be strictly greater than start_freq_hz.")
    return np.linspace(
        start_freq_hz,
        end_freq_hz,
        num=bin_count,
        endpoint=False,
        dtype=np.float64,
    )


def _resample_frame_to_grid(
    frame: np.ndarray,
    *,
    source_grid_hz: np.ndarray,
    target_grid_hz: np.ndarray,
) -> np.ndarray:
    """Resample one PSD frame onto the shared target frequency grid."""
    if frame.shape != source_grid_hz.shape:
        raise CodecConfigurationError("frame and source_grid_hz must have the same shape.")
    if target_grid_hz.ndim != 1 or target_grid_hz.size == 0:
        raise CodecConfigurationError("target_grid_hz must be a non-empty one-dimensional array.")
    if source_grid_hz.shape == target_grid_hz.shape and np.allclose(source_grid_hz, target_grid_hz):
        return frame.astype(np.float64, copy=False)
    return cast(
        np.ndarray,
        np.interp(target_grid_hz, source_grid_hz, frame).astype(np.float64, copy=False),
    )


def _estimate_sequence_noise_floors(
    frames: np.ndarray,
    *,
    window_length: int,
    percentile: float,
) -> np.ndarray:
    """Estimate reference noise floors inside one campaign/node time sequence."""
    noise_floors = np.empty_like(frames)
    for frame_index in range(frames.shape[0]):
        start_index = max(0, frame_index - window_length + 1)
        noise_floors[frame_index] = estimate_reference_noise_floor(
            frames[start_index : frame_index + 1],
            percentile=percentile,
        )
    return noise_floors


def _build_campaign_bundle(
    *,
    frames: list[np.ndarray],
    frequency_grid_hz: np.ndarray | None,
    noise_floors: list[np.ndarray] | None,
    campaign_labels: list[str],
    campaign_ids: list[int],
    node_labels: list[str],
    sequence_ids: list[str],
    timestamps_ms: list[int],
) -> CampaignDatasetBundle:
    """Assemble the final immutable campaign bundle."""
    if not frames or frequency_grid_hz is None:
        raise CodecConfigurationError("No PSD frames were loaded from the selected campaigns.")
    return CampaignDatasetBundle(
        frames=np.stack(frames, axis=0),
        frequency_grid_hz=frequency_grid_hz,
        noise_floors=None if noise_floors is None else np.stack(noise_floors, axis=0),
        campaign_labels=np.asarray(campaign_labels, dtype=np.str_),
        campaign_ids=np.asarray(campaign_ids, dtype=np.int64),
        node_labels=np.asarray(node_labels, dtype=np.str_),
        sequence_ids=np.asarray(sequence_ids, dtype=np.str_),
        timestamps_ms=np.asarray(timestamps_ms, dtype=np.int64),
    )


def _matches_any(value: str, patterns: list[str]) -> bool:
    """Return whether a string matches at least one shell-style glob pattern."""
    return any(fnmatch(value, pattern) for pattern in patterns)
