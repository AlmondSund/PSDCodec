"""Integration tests for the full operational codec pipeline."""

from __future__ import annotations

import numpy as np

from codec.config import (
    CodecRuntimeConfig,
    FactorizedEntropyModelConfig,
    PreprocessingConfig,
    ScalarQuantizerConfig,
)
from interfaces.api import PsdCodecService
from models.reference import ReferenceLinearCodecModel
from objectives.distortion import IllustrativeTaskConfig


def _make_service() -> PsdCodecService:
    """Create a service whose learned model is exact on zero latent chunks."""
    config = CodecRuntimeConfig(
        preprocessing=PreprocessingConfig(
            reduced_bin_count=4,
            block_count=2,
            dynamic_range_offset=1.0e-6,
            stability_epsilon=1.0e-8,
            mean_quantizer=ScalarQuantizerConfig(-10.0, 10.0, 12),
            log_sigma_quantizer=ScalarQuantizerConfig(-20.0, 5.0, 12),
        ),
        entropy_model=FactorizedEntropyModelConfig(alphabet_size=3, precision_bits=10),
    )
    model = ReferenceLinearCodecModel.from_identity_chunking(
        reduced_bin_count=4,
        latent_vector_count=2,
        embedding_dim=2,
    )
    codebook = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    )
    probabilities = np.asarray([0.6, 0.2, 0.2], dtype=np.float64)
    return PsdCodecService.create(
        config, model=model, codebook=codebook, probabilities=probabilities
    )


def test_encode_decode_round_trip_matches_encode_time_reconstruction() -> None:
    """Packet decoding should reproduce the same frame reconstructed during encoding."""
    service = _make_service()
    frame = np.full(8, 4.0, dtype=np.float64)

    encoded = service.encode_frame(frame)
    decoded = service.decode_packet(encoded.packet_bytes)

    assert np.array_equal(decoded.indices, encoded.quantization.indices)
    assert np.allclose(decoded.reconstructed_frame, encoded.reconstructed_frame, atol=1.0e-10)
    assert np.allclose(encoded.reconstructed_frame, encoded.preprocessing_only_frame, atol=1.0e-10)
    assert encoded.operational_bit_count == encoded.packet.operational_bit_count


def test_runtime_evaluation_reports_task_distortion_when_requested() -> None:
    """The evaluation surface should expose optional task-aware distortion."""
    service = _make_service()
    frame = np.full(8, 4.0, dtype=np.float64)
    noise_floor = np.full(8, 1.0, dtype=np.float64)
    frequency_grid_hz = np.linspace(100.0, 107.0, num=8, dtype=np.float64)

    evaluation = service.evaluate_frame(
        frame,
        noise_floor=noise_floor,
        frequency_grid_hz=frequency_grid_hz,
        task_config=IllustrativeTaskConfig(occupancy_margin=0.5, smoothing_window_bins=3),
    )

    assert evaluation.distortion.psd_distortion >= 0.0
    assert evaluation.distortion.preprocessing_distortion >= 0.0
    assert evaluation.distortion.codec_distortion >= 0.0
    assert evaluation.distortion.task_distortion is not None
