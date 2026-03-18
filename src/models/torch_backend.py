"""Optional PyTorch backend for learning and ONNX export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised only when torch is unavailable
    torch = None
    nn = None


@dataclass(frozen=True)
class TorchCodecConfig:
    """Neural-network configuration for the PyTorch learned codec."""

    reduced_bin_count: int  # Preprocessed frame length N_r
    latent_vector_count: int  # Number of latent positions M
    embedding_dim: int  # Latent dimension d
    codebook_size: int  # Number of VQ codewords J
    hidden_dim: int = 256  # Width of the shared MLP hidden representation
    commitment_weight: float = 0.25  # β_com in the manuscript


if torch is not None:

    @dataclass(frozen=True)
    class TorchVectorQuantizationOutput:
        """Structured output of the PyTorch vector-quantizer module."""

        straight_through_latents: torch.Tensor  # STE latents used by the decoder
        quantized_latents: torch.Tensor  # Detached codebook vectors
        indices: torch.Tensor  # Selected codeword indices
        vq_loss: torch.Tensor  # Stabilization loss LVQ

    @dataclass(frozen=True)
    class TorchTrainingOutput:
        """Structured output of the full differentiable training-time codec."""

        reconstructed_normalized_frames: torch.Tensor  # Decoder output û_t in normalized space
        indices: torch.Tensor  # Discrete codeword indices for each latent position
        vq_loss: torch.Tensor  # Stabilization loss LVQ
        rate_bits: torch.Tensor  # Training-time rate proxy R_idx in bits per frame
        quantized_latents: torch.Tensor  # Quantized latent vectors after codebook lookup

    class TorchFactorizedEntropyModel(nn.Module):
        """Learned factorized categorical model q_xi for latent indices."""

        def __init__(self, alphabet_size: int) -> None:
            """Initialize a learnable categorical prior over codeword indices."""
            super().__init__()
            self.logits = nn.Parameter(torch.zeros(alphabet_size, dtype=torch.float32))

        def probabilities(self) -> torch.Tensor:
            """Return the normalized symbol probabilities."""
            return torch.softmax(self.logits, dim=0)

        def rate_bits(self, indices: torch.Tensor) -> torch.Tensor:
            """Return the per-frame rate proxy `-Σ log2 q(i_m)`."""
            log_probabilities = torch.log_softmax(self.logits, dim=0)
            gathered = log_probabilities[indices]
            return -torch.sum(gathered, dim=1) / torch.log(
                torch.tensor(2.0, device=indices.device, dtype=log_probabilities.dtype)
            )

    class TorchMlpEncoder(nn.Module):
        """Inference-time encoder E_θ producing latent vectors before VQ assignment."""

        def __init__(self, config: TorchCodecConfig) -> None:
            """Build the MLP encoder used during training and export."""
            super().__init__()
            latent_width = config.latent_vector_count * config.embedding_dim
            self.reduced_bin_count = config.reduced_bin_count
            self.latent_vector_count = config.latent_vector_count
            self.embedding_dim = config.embedding_dim
            self.network = nn.Sequential(
                nn.Linear(config.reduced_bin_count, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, latent_width),
            )

        def forward(self, normalized_frames: torch.Tensor) -> torch.Tensor:
            """Encode batched normalized frames into `[batch, M, d]` latents."""
            latents = self.network(normalized_frames)
            return latents.view(-1, self.latent_vector_count, self.embedding_dim)

    class TorchMlpDecoder(nn.Module):
        """Decoder G_φ reconstructing normalized frames from quantized latents."""

        def __init__(self, config: TorchCodecConfig) -> None:
            """Build the MLP decoder."""
            super().__init__()
            latent_width = config.latent_vector_count * config.embedding_dim
            self.network = nn.Sequential(
                nn.Linear(latent_width, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.reduced_bin_count),
            )

        def forward(self, quantized_latents: torch.Tensor) -> torch.Tensor:
            """Decode batched latent tensors into normalized frames with length N_r."""
            latent_flat = quantized_latents.reshape(quantized_latents.shape[0], -1)
            return self.network(latent_flat)

    class TorchVectorQuantizer(nn.Module):
        """Codebook lookup with straight-through gradient estimation."""

        def __init__(
            self,
            *,
            codebook_size: int,  # Number of codewords J
            embedding_dim: int,  # Codeword dimension d
            commitment_weight: float,  # β_com in LVQ
        ) -> None:
            """Initialize the learnable codebook."""
            super().__init__()
            self.commitment_weight = commitment_weight
            self.codebook = nn.Parameter(torch.randn(codebook_size, embedding_dim) * 0.05)

        def forward(self, latents: torch.Tensor) -> TorchVectorQuantizationOutput:
            """Quantize latents and compute the stabilization loss."""
            batch_size, latent_vector_count, embedding_dim = latents.shape
            flat_latents = latents.reshape(-1, embedding_dim)
            distances = torch.sum(
                (flat_latents[:, None, :] - self.codebook[None, :, :]) ** 2,
                dim=2,
            )
            indices = torch.argmin(distances, dim=1)
            quantized = self.codebook[indices].reshape(
                batch_size, latent_vector_count, embedding_dim
            )
            straight_through = latents + (quantized - latents).detach()
            vq_loss = torch.mean(
                (latents.detach() - quantized) ** 2
            ) + self.commitment_weight * torch.mean(
                (latents - quantized.detach()) ** 2,
            )
            return TorchVectorQuantizationOutput(
                straight_through_latents=straight_through,
                quantized_latents=quantized,
                indices=indices.reshape(batch_size, latent_vector_count),
                vq_loss=vq_loss,
            )

    class TorchFullCodec(nn.Module):
        """Full PyTorch learned codec used during research-time training and validation."""

        def __init__(
            self,
            config: TorchCodecConfig,
        ) -> None:
            """Construct encoder, vector quantizer, and decoder modules."""
            super().__init__()
            self.config = config
            self.encoder = TorchMlpEncoder(config)
            self.vector_quantizer = TorchVectorQuantizer(
                codebook_size=config.codebook_size,
                embedding_dim=config.embedding_dim,
                commitment_weight=config.commitment_weight,
            )
            self.decoder = TorchMlpDecoder(config)
            self.entropy_model = TorchFactorizedEntropyModel(config.codebook_size)

        def encode_pre_quantization(self, normalized_frames: torch.Tensor) -> torch.Tensor:
            """Return encoder outputs before nearest-codeword assignment."""
            return self.encoder(normalized_frames)

        def forward(
            self,
            normalized_frames: torch.Tensor,  # Batched standardized frames [batch, N_r]
        ) -> TorchTrainingOutput:
            """Run the full differentiable codec forward pass."""
            latents = self.encoder(normalized_frames)
            quantization = self.vector_quantizer(latents)
            reconstructed = self.decoder(quantization.straight_through_latents)
            rate_bits = self.entropy_model.rate_bits(quantization.indices)
            return TorchTrainingOutput(
                reconstructed_normalized_frames=reconstructed,
                indices=quantization.indices,
                vq_loss=quantization.vq_loss,
                rate_bits=rate_bits,
                quantized_latents=quantization.quantized_latents,
            )

        def export_runtime_codebook(self) -> Any:
            """Return the current VQ codebook as a NumPy array."""
            return self.vector_quantizer.codebook.detach().cpu().numpy()

        def export_runtime_probabilities(self) -> Any:
            """Return the learned factorized entropy probabilities as a NumPy array."""
            return self.entropy_model.probabilities().detach().cpu().numpy()

        def export_encoder_to_onnx(
            self,
            output_path: Path,  # Destination `.onnx` file
            *,
            batch_size: int = 1,  # Example batch size for tracing
            opset_version: int = 17,  # ONNX opset used by the export
        ) -> Path:
            """Export the inference-time encoder boundary used by constrained devices."""
            example = torch.zeros(batch_size, self.config.reduced_bin_count, dtype=torch.float32)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.onnx.export(
                self.encoder,
                example,
                output_path,
                export_params=True,
                opset_version=opset_version,
                dynamo=False,
                input_names=["normalized_frame"],
                output_names=["pre_quantization_latents"],
                dynamic_axes={
                    "normalized_frame": {0: "batch_size"},
                    "pre_quantization_latents": {0: "batch_size"},
                },
            )
            return output_path

else:  # pragma: no cover - exercised only when torch is unavailable

    class TorchVectorQuantizationOutput:
        """Placeholder type raised when PyTorch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Reject construction without PyTorch."""
            raise ImportError("PyTorch is required to use the torch_backend module.")

    class TorchMlpEncoder:
        """Placeholder class raised when PyTorch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Reject construction without PyTorch."""
            raise ImportError("PyTorch is required to use the torch_backend module.")

    class TorchMlpDecoder:
        """Placeholder class raised when PyTorch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Reject construction without PyTorch."""
            raise ImportError("PyTorch is required to use the torch_backend module.")

    class TorchVectorQuantizer:
        """Placeholder class raised when PyTorch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Reject construction without PyTorch."""
            raise ImportError("PyTorch is required to use the torch_backend module.")

    class TorchFullCodec:
        """Placeholder class raised when PyTorch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Reject construction without PyTorch."""
            raise ImportError("PyTorch is required to use the torch_backend module.")
