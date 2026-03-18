"""Unit tests for side-information and packet serialization."""

from __future__ import annotations

import numpy as np

from codec.config import PacketFormatConfig, ScalarQuantizerConfig
from codec.packetization import PacketSerializer
from codec.types import CodecPacket, QuantizedSideInformation


def test_side_information_bit_packing_round_trip_recovers_codes() -> None:
    """Bit packing should preserve side-information codes exactly."""
    mean_quantizer = ScalarQuantizerConfig(-5.0, 5.0, 5)
    log_sigma_quantizer = ScalarQuantizerConfig(-8.0, 2.0, 4)
    serializer = PacketSerializer(PacketFormatConfig(), mean_quantizer, log_sigma_quantizer)
    side_information = QuantizedSideInformation(
        mean_codes=np.asarray([3, 12, 7], dtype=np.int64),
        log_sigma_codes=np.asarray([4, 9, 2], dtype=np.int64),
        means=np.asarray([-4.0322580645, -1.1290322581, -2.7419354839], dtype=np.float64),
        log_sigmas=np.asarray([-5.3333333333, -2.0, -6.6666666667], dtype=np.float64),
    )

    payload, bit_count = serializer.pack_side_information(side_information)
    unpacked = serializer.unpack_side_information(payload, bit_count, block_count=3)

    assert np.array_equal(unpacked.mean_codes, side_information.mean_codes)
    assert np.array_equal(unpacked.log_sigma_codes, side_information.log_sigma_codes)


def test_packet_serialization_round_trip_preserves_metadata_and_payloads() -> None:
    """Packet container serialization should be exact."""
    serializer = PacketSerializer(
        PacketFormatConfig(),
        ScalarQuantizerConfig(-5.0, 5.0, 5),
        ScalarQuantizerConfig(-8.0, 2.0, 4),
    )
    packet = CodecPacket(
        original_bin_count=16,
        reduced_bin_count=8,
        block_count=2,
        latent_vector_count=4,
        side_information_payload=b"\xa0",
        side_information_bit_count=5,
        index_payload=b"\x11\x22",
        index_bit_count=13,
    )

    serialized = serializer.serialize_packet(packet)
    restored = serializer.deserialize_packet(serialized)

    assert restored == packet
