"""Interfaces for model components that map normalized frames to latent vectors and back."""

from __future__ import annotations

from typing import Protocol

from utils import FloatArray


class LatentCodecModel(Protocol):
    """Inference-time model contract used by the operational codec pipeline."""

    @property
    def reduced_bin_count(self) -> int:
        """Return the preprocessed frame length N_r."""

    @property
    def latent_vector_count(self) -> int:
        """Return the number of latent positions M."""

    @property
    def embedding_dim(self) -> int:
        """Return the latent embedding dimension d."""

    def encode(
        self,
        normalized_frame: FloatArray,  # Standardized one-dimensional frame u_t
    ) -> FloatArray:
        """Map one normalized frame to a latent matrix with shape [M, d]."""

    def decode(
        self,
        quantized_latents: FloatArray,  # Quantized latent matrix with shape [M, d]
    ) -> FloatArray:
        """Map one quantized latent matrix back to a normalized frame with length N_r."""
