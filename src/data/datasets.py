"""Dataset contracts and preprocessing-aware PSD training data preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codec.preprocessing import FramePreprocessor
from codec.types import PreprocessingArtifacts
from data.campaigns import CampaignDatasetBundle, load_campaign_dataset_bundle
from objectives.distortion import estimate_reference_noise_floor
from utils import FloatArray


@dataclass(frozen=True)
class PreparedPsdSample:
    """One preprocessed PSD sample ready for training or validation."""

    original_frame: FloatArray  # Original PSD frame s_t
    normalized_frame: FloatArray  # Deterministically preprocessed frame u_t
    side_means: FloatArray  # Quantized/dequantized block means \hat{\mu}_b
    side_log_sigmas: (
        FloatArray  # Quantized/dequantized block log standard deviations \log \hat{\sigma}_b
    )
    noise_floor: FloatArray | None = None  # Optional reference noise floor \bar{n}_t


@dataclass(frozen=True)
class PreparedPsdBatch:
    """Batched PSD samples stacked for a training step."""

    original_frames: FloatArray  # Batch of original PSD frames with shape [batch, N]
    normalized_frames: FloatArray  # Batch of normalized frames with shape [batch, N_r]
    side_means: FloatArray  # Batch of block means with shape [batch, B]
    side_log_sigmas: FloatArray  # Batch of block log standard deviations with shape [batch, B]
    noise_floors: FloatArray | None = None  # Optional batch of noise floors with shape [batch, N]


@dataclass(frozen=True)
class PreparedPsdDataset:
    """Prepared PSD dataset with deterministic preprocessing applied once up front."""

    original_frames: np.ndarray  # Array of shape [num_frames, N]
    normalized_frames: np.ndarray  # Array of shape [num_frames, N_r]
    side_means: np.ndarray  # Array of shape [num_frames, B]
    side_log_sigmas: np.ndarray  # Array of shape [num_frames, B]
    frequency_grid_hz: FloatArray | None = None  # Optional shared frequency grid
    noise_floors: np.ndarray | None = None  # Optional array of shape [num_frames, N]

    @classmethod
    def from_frames(
        cls,
        frames: np.ndarray,  # Raw PSD frames with shape [num_frames, N]
        *,
        preprocessor: FramePreprocessor,
        frequency_grid_hz: FloatArray | None = None,
        noise_floors: np.ndarray | None = None,
        noise_floor_window: int | None = None,
        noise_floor_percentile: float = 10.0,
    ) -> PreparedPsdDataset:
        """Create a prepared dataset from an in-memory frame matrix."""
        frame_matrix = np.asarray(frames, dtype=np.float64)
        if frame_matrix.ndim != 2:
            raise ValueError("frames must have shape [num_frames, bin_count].")
        if frame_matrix.shape[0] < 1:
            raise ValueError("frames must contain at least one PSD frame.")

        if frequency_grid_hz is not None:
            frequency_grid = np.asarray(frequency_grid_hz, dtype=np.float64)
            if frequency_grid.shape != (frame_matrix.shape[1],):
                raise ValueError("frequency_grid_hz must have shape [bin_count].")
        else:
            frequency_grid = None

        resolved_noise_floors = _resolve_noise_floors(
            frame_matrix,
            explicit_noise_floors=noise_floors,
            noise_floor_window=noise_floor_window,
            noise_floor_percentile=noise_floor_percentile,
        )

        artifacts = [preprocessor.preprocess(frame) for frame in frame_matrix]
        return cls(
            original_frames=frame_matrix,
            normalized_frames=np.stack(
                [artifact.normalized_frame for artifact in artifacts], axis=0
            ),
            side_means=np.stack(
                [artifact.side_information.means for artifact in artifacts], axis=0
            ),
            side_log_sigmas=np.stack(
                [artifact.side_information.log_sigmas for artifact in artifacts],
                axis=0,
            ),
            frequency_grid_hz=frequency_grid,
            noise_floors=resolved_noise_floors,
        )

    @classmethod
    def from_npz(
        cls,
        dataset_path: str | Path,  # Input `.npz` file containing PSD arrays
        *,
        preprocessor: FramePreprocessor,
        frames_key: str = "frames",
        frequency_grid_key: str | None = "frequency_grid_hz",
        noise_floor_key: str | None = None,
        noise_floor_window: int | None = None,
        noise_floor_percentile: float = 10.0,
    ) -> PreparedPsdDataset:
        """Load a prepared PSD dataset from an `.npz` archive."""
        path = Path(dataset_path)
        with np.load(path, allow_pickle=False) as data:
            frames = data[frames_key]
            frequency_grid = (
                data[frequency_grid_key]
                if frequency_grid_key and frequency_grid_key in data
                else None
            )
            noise_floors = (
                data[noise_floor_key] if noise_floor_key and noise_floor_key in data else None
            )
        return cls.from_frames(
            frames,
            preprocessor=preprocessor,
            frequency_grid_hz=frequency_grid,
            noise_floors=noise_floors,
            noise_floor_window=noise_floor_window,
            noise_floor_percentile=noise_floor_percentile,
        )

    @classmethod
    def from_campaigns(
        cls,
        campaign_root: str | Path,  # Root directory containing raw campaign subdirectories
        *,
        preprocessor: FramePreprocessor,
        include_campaign_globs: list[str] | tuple[str, ...] | None = None,
        exclude_campaign_globs: list[str] | tuple[str, ...] | None = None,
        include_node_globs: list[str] | tuple[str, ...] | None = None,
        target_bin_count: int | None = None,
        value_scale: str = "db_to_power",
        max_frames: int | None = None,
        noise_floor_window: int | None = None,
        noise_floor_percentile: float = 10.0,
    ) -> PreparedPsdDataset:
        """Create a prepared dataset directly from raw campaign CSV acquisitions.

        The raw campaigns contain PSD vectors encoded inside CSV rows and may mix
        different RBW-dependent PSD lengths. This constructor delegates that boundary
        handling to `data.campaigns`, then applies the repository preprocessing once.
        """
        bundle = load_campaign_dataset_bundle(
            campaign_root,
            include_campaign_globs=include_campaign_globs,
            exclude_campaign_globs=exclude_campaign_globs,
            include_node_globs=include_node_globs,
            target_bin_count=target_bin_count,
            value_scale=value_scale,
            max_frames=max_frames,
            noise_floor_window=noise_floor_window,
            noise_floor_percentile=noise_floor_percentile,
        )
        return cls.from_campaign_bundle(bundle, preprocessor=preprocessor)

    @classmethod
    def from_campaign_bundle(
        cls,
        bundle: CampaignDatasetBundle,
        *,
        preprocessor: FramePreprocessor,
    ) -> PreparedPsdDataset:
        """Create a prepared dataset from a harmonized raw-campaign bundle."""
        return cls.from_frames(
            bundle.frames,
            preprocessor=preprocessor,
            frequency_grid_hz=bundle.frequency_grid_hz,
            noise_floors=bundle.noise_floors,
            noise_floor_window=None,
        )

    def __len__(self) -> int:
        """Return the number of frames in the dataset."""
        return int(self.original_frames.shape[0])

    def __getitem__(self, index: int) -> PreparedPsdSample:
        """Return one prepared sample by index."""
        noise_floor = None if self.noise_floors is None else self.noise_floors[index]
        return PreparedPsdSample(
            original_frame=self.original_frames[index],
            normalized_frame=self.normalized_frames[index],
            side_means=self.side_means[index],
            side_log_sigmas=self.side_log_sigmas[index],
            noise_floor=noise_floor,
        )

    @property
    def original_bin_count(self) -> int:
        """Return the original PSD frame length N."""
        return int(self.original_frames.shape[1])

    @property
    def reduced_bin_count(self) -> int:
        """Return the normalized frame length N_r."""
        return int(self.normalized_frames.shape[1])

    @property
    def block_count(self) -> int:
        """Return the number of standardization blocks B."""
        return int(self.side_means.shape[1])

    def subset(
        self,
        indices: np.ndarray,  # Selected frame indices
    ) -> PreparedPsdDataset:
        """Return a view-like dataset subset backed by copied NumPy arrays."""
        selected = np.asarray(indices, dtype=np.int64)
        noise_floors = None if self.noise_floors is None else self.noise_floors[selected]
        frequency_grid = None if self.frequency_grid_hz is None else self.frequency_grid_hz.copy()
        return PreparedPsdDataset(
            original_frames=self.original_frames[selected],
            normalized_frames=self.normalized_frames[selected],
            side_means=self.side_means[selected],
            side_log_sigmas=self.side_log_sigmas[selected],
            frequency_grid_hz=frequency_grid,
            noise_floors=noise_floors,
        )

    def train_validation_split(
        self,
        *,
        validation_fraction: float,
        seed: int = 0,
        shuffle: bool = True,
    ) -> tuple[PreparedPsdDataset, PreparedPsdDataset]:
        """Split the dataset into train and validation subsets."""
        if not (0.0 < validation_fraction < 1.0):
            raise ValueError("validation_fraction must lie in the open interval (0, 1).")
        indices = np.arange(len(self), dtype=np.int64)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        validation_count = max(1, int(round(validation_fraction * len(self))))
        training_count = len(self) - validation_count
        if training_count < 1:
            raise ValueError("validation_fraction leaves no training samples.")
        return self.subset(indices[:training_count]), self.subset(indices[training_count:])


def collate_prepared_psd_samples(samples: list[PreparedPsdSample]) -> PreparedPsdBatch:
    """Stack individual prepared samples into a batch dataclass."""
    original_frames = np.stack([sample.original_frame for sample in samples], axis=0)
    normalized_frames = np.stack([sample.normalized_frame for sample in samples], axis=0)
    side_means = np.stack([sample.side_means for sample in samples], axis=0)
    side_log_sigmas = np.stack([sample.side_log_sigmas for sample in samples], axis=0)
    if any(sample.noise_floor is None for sample in samples):
        noise_floors = None
    else:
        noise_floors = np.stack(
            [sample.noise_floor for sample in samples if sample.noise_floor is not None], axis=0
        )
    return PreparedPsdBatch(
        original_frames=original_frames,
        normalized_frames=normalized_frames,
        side_means=side_means,
        side_log_sigmas=side_log_sigmas,
        noise_floors=noise_floors,
    )


def preprocess_artifacts_to_sample(
    original_frame: FloatArray,
    artifacts: PreprocessingArtifacts,
    *,
    noise_floor: FloatArray | None = None,
) -> PreparedPsdSample:
    """Convert preprocessing artifacts into a dataset sample."""
    return PreparedPsdSample(
        original_frame=original_frame,
        normalized_frame=artifacts.normalized_frame,
        side_means=artifacts.side_information.means,
        side_log_sigmas=artifacts.side_information.log_sigmas,
        noise_floor=noise_floor,
    )


def _resolve_noise_floors(
    frames: np.ndarray,
    *,
    explicit_noise_floors: np.ndarray | None,
    noise_floor_window: int | None,
    noise_floor_percentile: float,
) -> np.ndarray | None:
    """Return explicit or history-estimated noise floors aligned with the frame matrix."""
    if explicit_noise_floors is not None:
        noise_floors = np.asarray(explicit_noise_floors, dtype=np.float64)
        if noise_floors.shape != frames.shape:
            raise ValueError("explicit noise_floors must have the same shape as frames.")
        return noise_floors
    if noise_floor_window is None:
        return None
    if noise_floor_window <= 0:
        raise ValueError("noise_floor_window must be strictly positive.")

    resolved = np.empty_like(frames)
    for frame_index in range(frames.shape[0]):
        start_index = max(0, frame_index - noise_floor_window + 1)
        resolved[frame_index] = estimate_reference_noise_floor(
            frames[start_index : frame_index + 1],
            percentile=noise_floor_percentile,
        )
    return resolved
