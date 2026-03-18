"""Reference NumPy inference model used by tests and non-PyTorch workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from codec.exceptions import CodecConfigurationError
from utils import FloatArray, as_1d_float_array

ActivationName = Literal["identity", "tanh"]


@dataclass(frozen=True)
class ReferenceLinearCodecModel:
    """Simple linear encoder/decoder pair with explicit latent reshaping."""

    encoder_matrix: FloatArray  # Matrix with shape [M*d, N_r]
    decoder_matrix: FloatArray  # Matrix with shape [N_r, M*d]
    encoder_bias: FloatArray  # Bias vector with shape [M*d]
    decoder_bias: FloatArray  # Bias vector with shape [N_r]
    latent_vector_count: int  # Number of latent positions M
    embedding_dim: int  # Latent dimension d
    activation: ActivationName = "identity"  # Nonlinearity applied symmetrically in encode/decode

    def __post_init__(self) -> None:
        """Validate tensor shapes and supported activation names."""
        encoder_matrix = np.asarray(self.encoder_matrix, dtype=np.float64)
        decoder_matrix = np.asarray(self.decoder_matrix, dtype=np.float64)
        encoder_bias = np.asarray(self.encoder_bias, dtype=np.float64)
        decoder_bias = np.asarray(self.decoder_bias, dtype=np.float64)
        latent_width = self.latent_vector_count * self.embedding_dim

        if self.latent_vector_count <= 0 or self.embedding_dim <= 0:
            raise CodecConfigurationError("latent_vector_count and embedding_dim must be positive.")
        if encoder_matrix.ndim != 2 or encoder_matrix.shape[0] != latent_width:
            raise CodecConfigurationError("encoder_matrix must have shape [M*d, N_r].")
        if decoder_matrix.ndim != 2 or decoder_matrix.shape[1] != latent_width:
            raise CodecConfigurationError("decoder_matrix must have shape [N_r, M*d].")
        if encoder_bias.shape != (latent_width,):
            raise CodecConfigurationError("encoder_bias must have shape [M*d].")
        if decoder_bias.shape != (decoder_matrix.shape[0],):
            raise CodecConfigurationError("decoder_bias must have shape [N_r].")
        if self.activation not in {"identity", "tanh"}:
            raise CodecConfigurationError("activation must be either 'identity' or 'tanh'.")

        object.__setattr__(self, "encoder_matrix", encoder_matrix)
        object.__setattr__(self, "decoder_matrix", decoder_matrix)
        object.__setattr__(self, "encoder_bias", encoder_bias)
        object.__setattr__(self, "decoder_bias", decoder_bias)

    @classmethod
    def from_identity_chunking(
        cls,
        *,
        reduced_bin_count: int,  # Preprocessed frame length N_r
        latent_vector_count: int,  # Number of latent positions M
        embedding_dim: int,  # Latent embedding dimension d
    ) -> ReferenceLinearCodecModel:
        """Create an exact identity model when `M * d == N_r`.

        This is useful for testing the deterministic codec logic independently from learned weights.
        """
        latent_width = latent_vector_count * embedding_dim
        if latent_width != reduced_bin_count:
            raise CodecConfigurationError(
                "Identity chunking requires latent_vector_count * embedding_dim "
                "== reduced_bin_count.",
            )
        identity = np.eye(reduced_bin_count, dtype=np.float64)
        zeros = np.zeros(reduced_bin_count, dtype=np.float64)
        return cls(
            encoder_matrix=identity,
            decoder_matrix=identity,
            encoder_bias=zeros,
            decoder_bias=zeros,
            latent_vector_count=latent_vector_count,
            embedding_dim=embedding_dim,
        )

    @property
    def reduced_bin_count(self) -> int:
        """Return the preprocessed frame length N_r."""
        return int(self.decoder_matrix.shape[0])

    def encode(
        self,
        normalized_frame: FloatArray,  # Standardized input frame u_t
    ) -> FloatArray:
        """Encode one normalized frame into an [M, d] latent matrix."""
        frame = as_1d_float_array(normalized_frame, name="normalized_frame", allow_negative=True)
        if frame.size != self.reduced_bin_count:
            raise CodecConfigurationError(
                "normalized_frame length does not match the model input size."
            )
        latent_flat = self.encoder_matrix @ frame + self.encoder_bias
        latent_flat = self._apply_activation(latent_flat)
        return latent_flat.reshape(self.latent_vector_count, self.embedding_dim)

    def decode(
        self,
        quantized_latents: FloatArray,  # Quantized latent matrix [M, d]
    ) -> FloatArray:
        """Decode one latent matrix into a normalized frame with length N_r."""
        latent_matrix = np.asarray(quantized_latents, dtype=np.float64)
        if latent_matrix.shape != (self.latent_vector_count, self.embedding_dim):
            raise CodecConfigurationError(
                "quantized_latents shape does not match the model latent shape."
            )
        latent_flat = latent_matrix.reshape(-1)
        latent_flat = self._apply_activation(latent_flat)
        return self.decoder_matrix @ latent_flat + self.decoder_bias

    def _apply_activation(
        self,
        values: FloatArray,  # Linear layer output before or after latent transport
    ) -> FloatArray:
        """Apply the configured elementwise activation."""
        if self.activation == "identity":
            return values
        return np.tanh(values)
