"""Public operational API for PSDCodec encode/decode/evaluate workflows."""

from __future__ import annotations

from dataclasses import dataclass

from codec.config import CodecRuntimeConfig
from codec.types import CodecDecodeResult, CodecEncodeResult
from models.base import LatentCodecModel
from objectives.distortion import IllustrativeTaskConfig
from pipelines.runtime import CodecEvaluation, OperationalCodec
from utils import FloatArray


@dataclass
class PsdCodecService:
    """Facade that exposes the operational codec behind a stable interface boundary."""

    runtime: OperationalCodec  # Application-layer codec orchestrator

    @classmethod
    def create(
        cls,
        config: CodecRuntimeConfig,  # Shared runtime configuration
        *,
        model: LatentCodecModel,  # Inference model used by the runtime
        codebook: FloatArray,  # Codebook matrix used for nearest-neighbor assignment
        probabilities: FloatArray | None = None,  # Optional entropy-model PMF
    ) -> PsdCodecService:
        """Create a service from config, model, and codebook assets."""
        return cls(
            runtime=OperationalCodec.from_runtime_config(
                config,
                model=model,
                codebook=codebook,
                probabilities=probabilities,
            ),
        )

    def encode_frame(
        self,
        frame: FloatArray,  # Original PSD frame s_t
    ) -> CodecEncodeResult:
        """Encode one frame into an operational codec packet."""
        return self.runtime.encode(frame)

    def decode_packet(
        self,
        packet_bytes: bytes,  # Serialized packet produced by `encode_frame`
    ) -> CodecDecodeResult:
        """Decode one packet back into a PSD frame."""
        return self.runtime.decode(packet_bytes)

    def evaluate_frame(
        self,
        frame: FloatArray,  # Original PSD frame s_t
        *,
        noise_floor: FloatArray | None = None,  # Optional reference baseline \bar{n}_t
        frequency_grid_hz: FloatArray | None = None,  # Optional frequency support ω_n
        task_config: IllustrativeTaskConfig | None = None,  # Optional D_task configuration
    ) -> CodecEvaluation:
        """Encode and evaluate one frame with optional task-aware distortion metrics."""
        return self.runtime.evaluate(
            frame,
            noise_floor=noise_floor,
            frequency_grid_hz=frequency_grid_hz,
            task_config=task_config,
        )
