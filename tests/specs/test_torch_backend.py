"""Unit tests for the PyTorch learned codec backend."""

from __future__ import annotations

import pytest

from codec.exceptions import CodecConfigurationError
from models.torch_backend import TorchCodecConfig, TorchFullCodec


def test_torch_codec_config_rejects_invalid_local_backbone_hyperparameters() -> None:
    """The convolutional codec config should validate local-backbone parameters."""
    with pytest.raises(CodecConfigurationError, match="residual_block_count"):
        TorchCodecConfig(
            reduced_bin_count=16,
            latent_vector_count=4,
            embedding_dim=2,
            codebook_size=8,
            residual_block_count=0,
        )

    with pytest.raises(CodecConfigurationError, match="convolution_kernel_size"):
        TorchCodecConfig(
            reduced_bin_count=16,
            latent_vector_count=4,
            embedding_dim=2,
            codebook_size=8,
            convolution_kernel_size=4,
        )


def test_torch_full_codec_preserves_public_shape_contract() -> None:
    """The convolutional backend must keep the training/output tensor contract stable."""
    torch = pytest.importorskip("torch")
    config = TorchCodecConfig(
        reduced_bin_count=16,
        latent_vector_count=5,
        embedding_dim=3,
        codebook_size=7,
        hidden_dim=12,
        residual_block_count=2,
        convolution_kernel_size=3,
    )
    model = TorchFullCodec(config)
    normalized_frames = torch.randn(4, config.reduced_bin_count, dtype=torch.float32)

    latents = model.encode_pre_quantization(normalized_frames)
    output = model(normalized_frames)

    assert latents.shape == (4, config.latent_vector_count, config.embedding_dim)
    assert output.reconstructed_normalized_frames.shape == (4, config.reduced_bin_count)
    assert output.indices.shape == (4, config.latent_vector_count)
    assert output.quantized_latents.shape == (4, config.latent_vector_count, config.embedding_dim)
    assert output.rate_bits.shape == (4,)
    assert output.vq_loss.shape == ()
