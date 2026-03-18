"""End-to-end orchestration for PSDCodec encode/decode/evaluate workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from codec.config import CodecRuntimeConfig
from codec.entropy import FactorizedEntropyCodec, FactorizedEntropyModel
from codec.exceptions import CodecConfigurationError
from codec.packetization import PacketSerializer
from codec.preprocessing import FramePreprocessor
from codec.quantization import VectorQuantizer
from codec.types import CodecDecodeResult, CodecEncodeResult, CodecPacket
from models.base import LatentCodecModel
from objectives.distortion import (
    DistortionBreakdown,
    IllustrativeTaskConfig,
    build_distortion_breakdown,
    illustrative_task_loss,
)
from utils import FloatArray, as_1d_float_array


@dataclass(frozen=True)
class CodecEvaluation:
    """Evaluation summary for one encoded frame."""

    encode_result: CodecEncodeResult  # Full encode-time artifacts
    distortion: DistortionBreakdown  # PSD and optional task distortion decomposition


@dataclass
class OperationalCodec:
    """Codec-first runtime combining deterministic logic with a latent inference model."""

    config: CodecRuntimeConfig  # Top-level runtime configuration
    model: LatentCodecModel  # Inference-time encoder/decoder model
    quantizer: VectorQuantizer  # VQ codebook used before entropy coding
    entropy_codec: FactorizedEntropyCodec  # Factorized arithmetic coding surface
    packet_serializer: PacketSerializer  # Container serializer for transport/storage
    preprocessor: FramePreprocessor  # Deterministic preprocessing chain

    @classmethod
    def from_runtime_config(
        cls,
        config: CodecRuntimeConfig,  # Top-level runtime configuration
        *,
        model: LatentCodecModel,  # Inference-time encoder/decoder model
        codebook: FloatArray,  # Codebook matrix with shape [J, d]
        probabilities: FloatArray | None = None,  # Optional explicit entropy PMF
    ) -> OperationalCodec:
        """Construct an operational codec from shared configuration and model assets."""
        entropy_model = FactorizedEntropyModel.from_config(
            config.entropy_model,
            probabilities=probabilities,
        )
        return cls(
            config=config,
            model=model,
            quantizer=VectorQuantizer(codebook=np.asarray(codebook, dtype=np.float64)),
            entropy_codec=FactorizedEntropyCodec(entropy_model),
            packet_serializer=PacketSerializer(
                config.packet_format,
                config.preprocessing.mean_quantizer,
                config.preprocessing.log_sigma_quantizer,
            ),
            preprocessor=FramePreprocessor(config.preprocessing),
        )

    def __post_init__(self) -> None:
        """Validate cross-component dimensional consistency."""
        if self.model.embedding_dim != self.quantizer.embedding_dim:
            raise CodecConfigurationError(
                "Model embedding_dim must match the codebook embedding dimension."
            )
        if self.model.latent_vector_count <= 0:
            raise CodecConfigurationError("Model latent_vector_count must be positive.")
        if self.quantizer.codeword_count != self.entropy_codec.model.alphabet_size:
            raise CodecConfigurationError(
                "Entropy model alphabet size must match the codebook size."
            )

    def encode(
        self,
        frame: FloatArray,  # Original PSD frame s_t
    ) -> CodecEncodeResult:
        """Encode one PSD frame into a packet and reconstruct it through the full pipeline."""
        original_frame = as_1d_float_array(frame, name="frame")
        expected_reduced_bins = self.config.preprocessing.resolve_reduced_bin_count(
            original_frame.size
        )
        if expected_reduced_bins != self.model.reduced_bin_count:
            raise CodecConfigurationError(
                "Preprocessing reduced_bin_count must match the inference model input size.",
            )

        preprocessing = self.preprocessor.preprocess(original_frame)
        latents = self.model.encode(preprocessing.normalized_frame)
        quantization = self.quantizer.quantize(latents)
        entropy_coding = self.entropy_codec.encode(quantization.indices)
        side_information_payload, side_information_bit_count = (
            self.packet_serializer.pack_side_information(
                preprocessing.side_information,
            )
        )

        packet = self.packet_serializer.deserialize_packet(
            self.packet_serializer.serialize_packet(
                packet=self._build_packet(
                    original_bin_count=original_frame.size,
                    reduced_bin_count=preprocessing.normalized_frame.size,
                    side_information_payload=side_information_payload,
                    side_information_bit_count=side_information_bit_count,
                    latent_vector_count=quantization.indices.size,
                    index_payload=entropy_coding.payload,
                    index_bit_count=entropy_coding.bit_count,
                ),
            ),
        )
        reconstructed_frame = self._decode_packet_object(packet)
        preprocessing_only_frame = self.preprocessor.inverse_preprocess(
            preprocessing.normalized_frame,
            preprocessing.side_information,
            original_bin_count=original_frame.size,
        )
        packet_bytes = self.packet_serializer.serialize_packet(packet)
        rate_proxy = (
            self.entropy_codec.model.rate_proxy(quantization.indices) + side_information_bit_count
        )

        return CodecEncodeResult(
            packet=packet,
            packet_bytes=packet_bytes,
            preprocessing=preprocessing,
            quantization=quantization,
            reconstructed_frame=reconstructed_frame,
            preprocessing_only_frame=preprocessing_only_frame,
            operational_bit_count=packet.operational_bit_count,
            rate_proxy_bit_count=rate_proxy,
        )

    def decode(
        self,
        packet_bytes: bytes,  # Serialized packet bytes produced by `encode`
    ) -> CodecDecodeResult:
        """Decode a previously encoded packet back into a PSD frame."""
        packet = self.packet_serializer.deserialize_packet(packet_bytes)
        indices, reconstructed_frame = self._decode_packet_with_indices(packet)
        return CodecDecodeResult(
            packet=packet,
            indices=indices,
            reconstructed_frame=reconstructed_frame,
        )

    def evaluate(
        self,
        frame: FloatArray,  # Original PSD frame s_t
        *,
        noise_floor: FloatArray | None = None,  # Optional reference noise floor \bar{n}_t
        frequency_grid_hz: FloatArray | None = None,  # Optional frequency support ω_n
        task_config: IllustrativeTaskConfig | None = None,  # Optional D_task configuration
    ) -> CodecEvaluation:
        """Encode one frame and compute the distortion decomposition."""
        encode_result = self.encode(frame)
        task_distortion: float | None = None
        if task_config is not None:
            if noise_floor is None or frequency_grid_hz is None:
                raise CodecConfigurationError(
                    "task_config requires both noise_floor and frequency_grid_hz.",
                )
            task_distortion = illustrative_task_loss(
                frame,
                encode_result.reconstructed_frame,
                noise_floor=noise_floor,
                frequency_grid_hz=frequency_grid_hz,
                config=task_config,
            )
        distortion = build_distortion_breakdown(
            frame,
            encode_result.preprocessing_only_frame,
            encode_result.reconstructed_frame,
            dynamic_range_offset=self.config.preprocessing.dynamic_range_offset,
            task_distortion=task_distortion,
        )
        return CodecEvaluation(encode_result=encode_result, distortion=distortion)

    def _decode_packet_object(
        self,
        packet: CodecPacket,  # Parsed codec packet
    ) -> FloatArray:
        """Decode a packet directly to a reconstructed frame."""
        _, reconstructed_frame = self._decode_packet_with_indices(packet)
        return reconstructed_frame

    def _decode_packet_with_indices(
        self,
        packet: CodecPacket,  # Parsed codec packet
    ) -> tuple[np.ndarray, FloatArray]:
        """Decode a packet into both codeword indices and the reconstructed frame."""
        side_information = self.packet_serializer.unpack_side_information(
            packet.side_information_payload,
            packet.side_information_bit_count,
            block_count=packet.block_count,
        )
        indices = self.entropy_codec.decode(
            packet.index_payload,
            packet.index_bit_count,
            symbol_count=packet.latent_vector_count,
        )
        quantized_latents = self.quantizer.decode(indices)
        reconstructed_normalized = self.model.decode(quantized_latents)
        reconstructed_frame = self.preprocessor.inverse_preprocess(
            reconstructed_normalized,
            side_information,
            original_bin_count=packet.original_bin_count,
        )
        return indices, reconstructed_frame

    def _build_packet(
        self,
        *,
        original_bin_count: int,
        reduced_bin_count: int,
        side_information_payload: bytes,
        side_information_bit_count: int,
        latent_vector_count: int,
        index_payload: bytes,
        index_bit_count: int,
    ) -> CodecPacket:
        """Assemble a `CodecPacket` from validated payload segments."""
        return CodecPacket(
            original_bin_count=original_bin_count,
            reduced_bin_count=reduced_bin_count,
            block_count=self.config.preprocessing.block_count,
            latent_vector_count=latent_vector_count,
            side_information_payload=side_information_payload,
            side_information_bit_count=side_information_bit_count,
            index_payload=index_payload,
            index_bit_count=index_bit_count,
        )
