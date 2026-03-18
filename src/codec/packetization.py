"""Bit packing and container serialization for encoded PSD frames."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from codec.arithmetic import BitInputStream, BitOutputStream
from codec.config import PacketFormatConfig, ScalarQuantizerConfig
from codec.exceptions import CodecDecodeError, CodecEncodeError
from codec.quantization import UniformScalarQuantizer
from codec.types import CodecPacket, QuantizedSideInformation

_HEADER_STRUCT = struct.Struct(">4sBIIIIII")


def _payload_byte_count(bit_count: int) -> int:
    """Return the number of bytes required to store a finite bit payload."""
    return (bit_count + 7) // 8


@dataclass
class PacketSerializer:
    """Serialize codec packets and exact-width side-information bit payloads."""

    format_config: PacketFormatConfig  # Container magic/version settings
    mean_quantizer_config: ScalarQuantizerConfig  # Bit width for μ_b
    log_sigma_quantizer_config: ScalarQuantizerConfig  # Bit width for log σ_b

    def __post_init__(self) -> None:
        """Construct shared quantizers for deterministic side-information reconstruction."""
        self._mean_quantizer = UniformScalarQuantizer(self.mean_quantizer_config)
        self._log_sigma_quantizer = UniformScalarQuantizer(self.log_sigma_quantizer_config)

    def pack_side_information(
        self,
        side_information: QuantizedSideInformation,  # Quantized block statistics
    ) -> tuple[bytes, int]:
        """Bit-pack mean and log-sigma codes using their configured widths."""
        if side_information.mean_codes.size != side_information.log_sigma_codes.size:
            raise CodecEncodeError("Side-information mean and log-sigma code counts must match.")

        writer = BitOutputStream()
        for mean_code, log_sigma_code in zip(
            side_information.mean_codes.tolist(),
            side_information.log_sigma_codes.tolist(),
            strict=True,
        ):
            self._write_fixed_width(writer, int(mean_code), self.mean_quantizer_config.bits)
            self._write_fixed_width(
                writer, int(log_sigma_code), self.log_sigma_quantizer_config.bits
            )
        payload = writer.finish()
        return payload, writer.bit_count

    def unpack_side_information(
        self,
        payload: bytes,  # Packed side-information bits
        bit_count: int,  # Number of meaningful bits in the payload
        *,
        block_count: int,  # Number of standardization blocks B
    ) -> QuantizedSideInformation:
        """Recover quantized side-information codes and reconstruction values from bits."""
        reader = BitInputStream(payload, bit_count)
        mean_codes = np.empty(block_count, dtype=np.int64)
        log_sigma_codes = np.empty(block_count, dtype=np.int64)
        for block_index in range(block_count):
            mean_codes[block_index] = self._read_fixed_width(
                reader, self.mean_quantizer_config.bits
            )
            log_sigma_codes[block_index] = self._read_fixed_width(
                reader,
                self.log_sigma_quantizer_config.bits,
            )
        return QuantizedSideInformation(
            mean_codes=mean_codes,
            log_sigma_codes=log_sigma_codes,
            means=self._mean_quantizer.dequantize(mean_codes),
            log_sigmas=self._log_sigma_quantizer.dequantize(log_sigma_codes),
        )

    def serialize_packet(
        self,
        packet: CodecPacket,  # Self-describing packet metadata and payloads
    ) -> bytes:
        """Serialize a packet into bytes suitable for persistence or transmission."""
        header = _HEADER_STRUCT.pack(
            self.format_config.magic,
            self.format_config.version,
            packet.original_bin_count,
            packet.reduced_bin_count,
            packet.block_count,
            packet.latent_vector_count,
            packet.side_information_bit_count,
            packet.index_bit_count,
        )
        return header + packet.side_information_payload + packet.index_payload

    def deserialize_packet(
        self,
        payload: bytes,  # Byte-aligned packet container
    ) -> CodecPacket:
        """Deserialize a packet previously produced by `serialize_packet`."""
        if len(payload) < _HEADER_STRUCT.size:
            raise CodecDecodeError("Packet payload is smaller than the fixed header.")
        (
            magic,
            version,
            original_bin_count,
            reduced_bin_count,
            block_count,
            latent_vector_count,
            side_information_bit_count,
            index_bit_count,
        ) = _HEADER_STRUCT.unpack(payload[: _HEADER_STRUCT.size])
        if magic != self.format_config.magic:
            raise CodecDecodeError("Packet magic does not match the PSDCodec format.")
        if version != self.format_config.version:
            raise CodecDecodeError("Packet version does not match the configured serializer.")

        side_payload_bytes = _payload_byte_count(side_information_bit_count)
        index_payload_bytes = _payload_byte_count(index_bit_count)
        expected_size = _HEADER_STRUCT.size + side_payload_bytes + index_payload_bytes
        if len(payload) != expected_size:
            raise CodecDecodeError(
                "Packet payload length does not match the encoded header lengths."
            )

        side_start = _HEADER_STRUCT.size
        side_stop = side_start + side_payload_bytes
        index_stop = side_stop + index_payload_bytes
        return CodecPacket(
            original_bin_count=original_bin_count,
            reduced_bin_count=reduced_bin_count,
            block_count=block_count,
            latent_vector_count=latent_vector_count,
            side_information_payload=payload[side_start:side_stop],
            side_information_bit_count=side_information_bit_count,
            index_payload=payload[side_stop:index_stop],
            index_bit_count=index_bit_count,
        )

    def _write_fixed_width(
        self,
        writer: BitOutputStream,  # Destination bit stream
        value: int,  # Unsigned integer to pack
        bit_width: int,  # Exact field width in bits
    ) -> None:
        """Write an unsigned integer using a fixed bit width."""
        if value < 0 or value >= (1 << bit_width):
            raise CodecEncodeError("Fixed-width field value exceeds the configured bit width.")
        for shift in range(bit_width - 1, -1, -1):
            writer.write((value >> shift) & 1)

    def _read_fixed_width(
        self,
        reader: BitInputStream,  # Source bit stream
        bit_width: int,  # Exact field width in bits
    ) -> int:
        """Read an unsigned integer encoded with a fixed bit width."""
        value = 0
        for _ in range(bit_width):
            value = (value << 1) | reader.read()
        return value
