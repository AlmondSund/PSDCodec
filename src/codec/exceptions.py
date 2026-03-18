"""Codec-specific exceptions with explicit boundary semantics."""


class CodecError(RuntimeError):
    """Base class for operational codec failures."""


class CodecConfigurationError(ValueError):
    """Raised when codec configuration is internally inconsistent."""


class CodecEncodeError(CodecError):
    """Raised when a frame cannot be encoded into a valid payload."""


class CodecDecodeError(CodecError):
    """Raised when a payload cannot be decoded into a valid frame."""
