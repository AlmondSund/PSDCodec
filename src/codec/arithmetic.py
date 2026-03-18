"""Reference integer arithmetic coder for discrete latent indices."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

from codec.exceptions import CodecConfigurationError, CodecDecodeError, CodecEncodeError


@dataclass(frozen=True)
class FrequencyTable:
    """Static integer frequency table for arithmetic coding."""

    counts: tuple[int, ...]  # Positive symbol counts whose sum defines the coding total
    _cumulative: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the frequency table and precompute its cumulative sum."""
        if len(self.counts) < 2:
            raise CodecEncodeError("Frequency table must contain at least two symbols.")
        if any(count <= 0 for count in self.counts):
            raise CodecEncodeError("Frequency table counts must be strictly positive.")
        cumulative = [0]
        running_total = 0
        for count in self.counts:
            running_total += count
            cumulative.append(running_total)
        object.__setattr__(self, "_cumulative", tuple(cumulative))

    @property
    def total(self) -> int:
        """Return the total frequency mass."""
        return self._cumulative[-1]

    @property
    def symbol_limit(self) -> int:
        """Return the number of representable symbols."""
        return len(self.counts)

    def low(self, symbol: int) -> int:
        """Return the cumulative low endpoint of a symbol interval."""
        return self._cumulative[symbol]

    def high(self, symbol: int) -> int:
        """Return the cumulative high endpoint of a symbol interval."""
        return self._cumulative[symbol + 1]

    def symbol_for_cumulative_value(self, value: int) -> int:
        """Return the symbol whose cumulative interval contains the given value."""
        symbol = bisect_right(self._cumulative, value) - 1
        if not (0 <= symbol < self.symbol_limit):
            raise CodecDecodeError("Arithmetic decoder produced an out-of-range cumulative value.")
        return symbol


class BitOutputStream:
    """Append individual bits to a byte-aligned payload."""

    def __init__(self) -> None:
        """Initialize an empty bit stream."""
        self._buffer = bytearray()
        self._current_byte = 0
        self._bit_offset = 0
        self.bit_count = 0

    def write(self, bit: int) -> None:
        """Append one bit to the output stream."""
        if bit not in (0, 1):
            raise CodecEncodeError("Arithmetic coder bits must be 0 or 1.")
        self._current_byte = (self._current_byte << 1) | bit
        self._bit_offset += 1
        self.bit_count += 1
        if self._bit_offset == 8:
            self._buffer.append(self._current_byte)
            self._current_byte = 0
            self._bit_offset = 0

    def finish(self) -> bytes:
        """Pad the final byte with zeros and return the payload bytes."""
        if self._bit_offset > 0:
            self._buffer.append(self._current_byte << (8 - self._bit_offset))
            self._current_byte = 0
            self._bit_offset = 0
        return bytes(self._buffer)


class BitInputStream:
    """Read individual bits from a byte-aligned payload."""

    def __init__(
        self,
        payload: bytes,  # Byte-aligned payload
        bit_count: int,  # Number of meaningful bits inside that payload
    ) -> None:
        """Initialize an input stream over a finite bit payload."""
        self._payload = payload
        self._bit_count = bit_count
        self._read_bits = 0

    def read(self) -> int:
        """Read one bit, returning zero after the coded payload is exhausted."""
        if self._read_bits >= self._bit_count:
            self._read_bits += 1
            return 0
        byte_index, bit_index = divmod(self._read_bits, 8)
        current_byte = self._payload[byte_index]
        self._read_bits += 1
        return (current_byte >> (7 - bit_index)) & 1


class _ArithmeticCoderBase:
    """Shared interval update logic for arithmetic encoding and decoding."""

    def __init__(self, *, num_state_bits: int = 32) -> None:
        """Initialize the coder state with a full-range interval."""
        if not (1 <= num_state_bits <= 62):
            raise CodecConfigurationError("num_state_bits must lie in [1, 62].")
        self.num_state_bits = num_state_bits
        self.full_range = 1 << num_state_bits
        self.half_range = self.full_range >> 1
        self.quarter_range = self.half_range >> 1
        self.state_mask = self.full_range - 1
        self.low = 0
        self.high = self.state_mask

    def update(self, frequencies: FrequencyTable, symbol: int) -> None:
        """Narrow the current interval using one symbol."""
        if not (0 <= symbol < frequencies.symbol_limit):
            raise CodecEncodeError("Arithmetic coder symbol is outside the configured alphabet.")
        if frequencies.total >= self.quarter_range:
            raise CodecEncodeError("Frequency table total is too large for the coder state size.")

        current_range = self.high - self.low + 1
        symbol_low = frequencies.low(symbol)
        symbol_high = frequencies.high(symbol)
        new_low = self.low + (symbol_low * current_range) // frequencies.total
        new_high = self.low + (symbol_high * current_range) // frequencies.total - 1
        self.low = new_low
        self.high = new_high

        while ((self.low ^ self.high) & self.half_range) == 0:
            self.shift()
            self.low = (self.low << 1) & self.state_mask
            self.high = ((self.high << 1) & self.state_mask) | 1

        while (self.low & ~self.high & self.quarter_range) != 0:
            self.underflow()
            self.low = (self.low << 1) ^ self.half_range
            self.high = ((self.high ^ self.half_range) << 1) | self.half_range | 1

    def shift(self) -> None:
        """Handle a stable leading bit after interval renormalization."""
        raise NotImplementedError

    def underflow(self) -> None:
        """Handle the arithmetic underflow condition."""
        raise NotImplementedError


class ArithmeticEncoder(_ArithmeticCoderBase):
    """Static arithmetic encoder producing a compact bitstream."""

    def __init__(self, *, num_state_bits: int = 32) -> None:
        """Initialize an encoder with an empty output stream."""
        super().__init__(num_state_bits=num_state_bits)
        self._pending_underflow_bits = 0
        self._output = BitOutputStream()

    def write(self, frequencies: FrequencyTable, symbol: int) -> None:
        """Encode one symbol under the supplied frequency table."""
        self.update(frequencies, symbol)

    def finish(self) -> tuple[bytes, int]:
        """Finalize the arithmetic stream and return bytes plus exact bit length."""
        self._pending_underflow_bits += 1
        if self.low < self.quarter_range:
            self._write_bit(0)
        else:
            self._write_bit(1)
        payload = self._output.finish()
        return payload, self._output.bit_count

    def shift(self) -> None:
        """Emit a stable leading bit plus any delayed underflow complements."""
        bit = self.low >> (self.num_state_bits - 1)
        self._write_bit(bit)

    def underflow(self) -> None:
        """Delay output until the ambiguous middle interval resolves."""
        self._pending_underflow_bits += 1

    def _write_bit(self, bit: int) -> None:
        """Write one resolved bit and flush pending complements."""
        self._output.write(bit)
        complement = bit ^ 1
        while self._pending_underflow_bits > 0:
            self._output.write(complement)
            self._pending_underflow_bits -= 1


class ArithmeticDecoder(_ArithmeticCoderBase):
    """Static arithmetic decoder consuming a finite bitstream."""

    def __init__(
        self,
        payload: bytes,  # Byte-aligned arithmetic bitstream
        bit_count: int,  # Number of meaningful bits inside the payload
        *,
        num_state_bits: int = 32,
    ) -> None:
        """Initialize a decoder and preload its arithmetic state."""
        super().__init__(num_state_bits=num_state_bits)
        self._input = BitInputStream(payload, bit_count)
        self.code = 0
        for _ in range(self.num_state_bits):
            self.code = ((self.code << 1) & self.state_mask) | self._input.read()

    def read(self, frequencies: FrequencyTable) -> int:
        """Decode one symbol under the supplied frequency table."""
        current_range = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * frequencies.total - 1) // current_range
        symbol = frequencies.symbol_for_cumulative_value(value)
        self.update(frequencies, symbol)
        return symbol

    def shift(self) -> None:
        """Consume a new leading bit after interval renormalization."""
        self.code = ((self.code << 1) & self.state_mask) | self._input.read()

    def underflow(self) -> None:
        """Consume one deferred underflow bit."""
        self.code = (
            (self.code & self.half_range)
            | ((self.code << 1) & (self.state_mask >> 1))
            | self._input.read()
        )
