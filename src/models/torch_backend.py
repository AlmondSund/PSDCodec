"""Optional PyTorch backend for learning and ONNX export."""

from __future__ import annotations

import copy
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codec.exceptions import CodecConfigurationError

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ImportError:  # pragma: no cover - exercised only when torch is unavailable
    torch = None
    functional = None
    nn = None


@dataclass(frozen=True)
class TorchCodecConfig:
    """Neural-network configuration for the PyTorch learned codec."""

    reduced_bin_count: int  # Preprocessed frame length N_r
    latent_vector_count: int  # Number of latent positions M
    embedding_dim: int  # Latent dimension d
    codebook_size: int  # Number of VQ codewords J
    hidden_dim: int = 256  # Width of the shared convolutional feature representation
    residual_block_count: int = 4  # Number of residual spectral blocks per encoder/decoder
    convolution_kernel_size: int = 7  # Odd local receptive field width in frequency bins
    commitment_weight: float = 0.25  # β_com in the manuscript

    def __post_init__(self) -> None:
        """Validate codec dimensions and local-backbone hyperparameters."""
        if self.reduced_bin_count <= 0:
            raise CodecConfigurationError("reduced_bin_count must be strictly positive.")
        if self.latent_vector_count <= 0:
            raise CodecConfigurationError("latent_vector_count must be strictly positive.")
        if self.embedding_dim <= 0:
            raise CodecConfigurationError("embedding_dim must be strictly positive.")
        if self.codebook_size <= 0:
            raise CodecConfigurationError("codebook_size must be strictly positive.")
        if self.hidden_dim <= 0:
            raise CodecConfigurationError("hidden_dim must be strictly positive.")
        if self.residual_block_count <= 0:
            raise CodecConfigurationError("residual_block_count must be strictly positive.")
        if self.convolution_kernel_size <= 0 or self.convolution_kernel_size % 2 == 0:
            raise CodecConfigurationError(
                "convolution_kernel_size must be a strictly positive odd integer.",
            )
        if self.commitment_weight < 0.0:
            raise CodecConfigurationError("commitment_weight must be non-negative.")


if torch is not None:

    def _resolve_group_norm_group_count(
        channel_count: int,
    ) -> int:
        """Return a valid `GroupNorm` group count that divides `channel_count`.

        Purpose:
            The convolutional backbone should behave consistently across small test
            models and larger demo configurations. This helper therefore chooses the
            largest reasonable group count up to eight that exactly divides the channel
            width, falling back to instance-like normalization when needed.
        """
        for candidate in range(min(8, channel_count), 0, -1):
            if channel_count % candidate == 0:
                return candidate
        return 1

    def _initialize_residual_skip_projection_to_zero(
        projection: nn.Conv1d,
    ) -> None:
        """Initialize a residual projection as a neutral zero contribution.

        Purpose:
            The encoder and decoder both use 1x1 skip projections around the learned
            convolutional path. Starting those projections at a random scale made the
            encoder emit latents that were tens of times larger than the VQ codebook,
            which in turn caused the quantizer loss to dominate training. A zero
            initialization keeps the residual path available without letting it drown
            the optimization signal at step zero.
        """
        nn.init.zeros_(projection.weight)
        if projection.bias is not None:
            nn.init.zeros_(projection.bias)

    def _initialize_bounded_codebook(
        codebook: nn.Parameter,
    ) -> None:
        """Initialize codewords on the same scale as the bounded encoder latents.

        Purpose:
            The encoder now constrains its pre-quantization activations with `tanh`,
            so the codebook should start in a comparable range instead of near zero.
            Matching the initial latent/codeword scale reduces the large Euclidean
            mismatch that previously made `vq_loss` dominate the objective.
        """
        embedding_dim = max(1, int(codebook.shape[1]))
        bound = float(embedding_dim) ** -0.5
        nn.init.uniform_(codebook, -bound, bound)

    class TorchResidualConvBlock(nn.Module):
        """Local residual block that preserves neighborhood structure in PSD space."""

        def __init__(
            self,
            channel_count: int,  # Number of feature channels carried by the block
            *,
            kernel_size: int,  # Odd convolutional receptive field width
        ) -> None:
            """Build one normalization-convolution residual block."""
            super().__init__()
            padding = kernel_size // 2
            group_count = _resolve_group_norm_group_count(channel_count)
            self.pre_norm = nn.GroupNorm(group_count, channel_count)
            self.mid_norm = nn.GroupNorm(group_count, channel_count)
            self.activation = nn.GELU()
            self.conv1 = nn.Conv1d(
                channel_count,
                channel_count,
                kernel_size=kernel_size,
                padding=padding,
            )
            self.conv2 = nn.Conv1d(
                channel_count,
                channel_count,
                kernel_size=kernel_size,
                padding=padding,
            )

        def forward(
            self,
            spectral_features: torch.Tensor,  # Batched local spectral features [batch, C, N]
        ) -> torch.Tensor:
            """Refine spectral features while preserving a residual identity path."""
            residual = spectral_features
            normalized = self.pre_norm(spectral_features)
            transformed = self.conv1(normalized)
            transformed = self.activation(transformed)
            transformed = self.mid_norm(transformed)
            transformed = self.conv2(transformed)
            return residual + transformed

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

    class TorchSpectralEncoder(nn.Module):
        """Convolutional encoder preserving local PSD structure before VQ assignment."""

        def __init__(self, config: TorchCodecConfig) -> None:
            """Build the locality-preserving encoder used during training and export.

            Purpose:
                Narrow spectral peaks are strongly local in frequency, so a flat MLP
                discards the most useful inductive bias for this domain. The encoder
                therefore operates on `[batch, 1, N_r]` feature maps, refines them
                through residual 1D convolutions, and only then pools them into the
                `M` latent positions consumed by vector quantization.
            """
            super().__init__()
            self.reduced_bin_count = config.reduced_bin_count
            self.latent_vector_count = config.latent_vector_count
            self.embedding_dim = config.embedding_dim
            padding = config.convolution_kernel_size // 2
            group_count = _resolve_group_norm_group_count(config.hidden_dim)
            self.input_projection = nn.Conv1d(
                1,
                config.hidden_dim,
                kernel_size=config.convolution_kernel_size,
                padding=padding,
            )
            self.stem_residual_block = TorchResidualConvBlock(
                config.hidden_dim,
                kernel_size=config.convolution_kernel_size,
            )
            self.latent_residual_blocks = nn.Sequential(
                *[
                    TorchResidualConvBlock(
                        config.hidden_dim,
                        kernel_size=config.convolution_kernel_size,
                    )
                    for _ in range(config.residual_block_count)
                ]
            )
            self.output_norm = nn.GroupNorm(group_count, config.hidden_dim)
            self.output_activation = nn.GELU()
            self.output_projection = nn.Conv1d(config.hidden_dim, config.embedding_dim, 1)
            self.input_skip_projection = nn.Conv1d(1, config.embedding_dim, 1)
            self.output_bounding = nn.Tanh()
            _initialize_residual_skip_projection_to_zero(self.input_skip_projection)

        def forward(self, normalized_frames: torch.Tensor) -> torch.Tensor:
            """Encode batched normalized frames into `[batch, M, d]` latents."""
            spectral_input = normalized_frames.unsqueeze(1)
            spectral_features = self.input_projection(spectral_input)
            spectral_features = self.stem_residual_block(spectral_features)
            spectral_features = functional.interpolate(
                spectral_features,
                size=self.latent_vector_count,
                mode="nearest",
            )
            spectral_features = self.latent_residual_blocks(spectral_features)
            spectral_features = self.output_norm(spectral_features)
            spectral_features = self.output_activation(spectral_features)
            latent_features = self.output_projection(spectral_features)
            latent_skip = functional.interpolate(
                self.input_skip_projection(spectral_input),
                size=self.latent_vector_count,
                mode="nearest",
            )

            # Bound the encoder output before vector quantization so the latent/codebook
            # distance scale stays well-conditioned across random initialization and
            # early training.
            bounded_latents = self.output_bounding(latent_features + latent_skip)
            return bounded_latents.transpose(1, 2).contiguous()

    class TorchSpectralDecoder(nn.Module):
        """Convolutional decoder reconstructing normalized frames from local latents."""

        def __init__(self, config: TorchCodecConfig) -> None:
            """Build the locality-preserving decoder.

            Purpose:
                The decoder first mixes information locally across neighboring latent
                positions, then interpolates those features back onto the full
                frequency grid. This keeps the reconstruction path aligned with the
                underlying PSD geometry instead of asking one dense layer to memorize
                every peak interaction globally.
            """
            super().__init__()
            padding = config.convolution_kernel_size // 2
            group_count = _resolve_group_norm_group_count(config.hidden_dim)
            self.reduced_bin_count = config.reduced_bin_count
            self.input_projection = nn.Conv1d(config.embedding_dim, config.hidden_dim, 1)
            self.output_skip_projection = nn.Conv1d(config.embedding_dim, 1, 1)
            self.latent_residual_blocks = nn.Sequential(
                *[
                    TorchResidualConvBlock(
                        config.hidden_dim,
                        kernel_size=config.convolution_kernel_size,
                    )
                    for _ in range(config.residual_block_count)
                ]
            )
            self.output_residual_block = TorchResidualConvBlock(
                config.hidden_dim,
                kernel_size=config.convolution_kernel_size,
            )
            self.output_norm = nn.GroupNorm(group_count, config.hidden_dim)
            self.output_activation = nn.GELU()
            self.output_projection = nn.Conv1d(
                config.hidden_dim,
                1,
                kernel_size=config.convolution_kernel_size,
                padding=padding,
            )
            _initialize_residual_skip_projection_to_zero(self.output_skip_projection)

        def forward(self, quantized_latents: torch.Tensor) -> torch.Tensor:
            """Decode batched latent tensors into normalized frames with length N_r."""
            latent_features = quantized_latents.transpose(1, 2).contiguous()
            spectral_features = self.input_projection(latent_features)
            spectral_features = self.latent_residual_blocks(spectral_features)

            # Expand the latent grid back to the reduced PSD resolution using nearest
            # interpolation so sharp narrow peaks are not blurred before local refinement.
            spectral_features = functional.interpolate(
                spectral_features,
                size=self.reduced_bin_count,
                mode="nearest",
            )
            spectral_features = self.output_residual_block(spectral_features)
            spectral_features = self.output_norm(spectral_features)
            spectral_features = self.output_activation(spectral_features)
            reconstructed = self.output_projection(spectral_features)
            reconstructed_skip = functional.interpolate(
                self.output_skip_projection(latent_features),
                size=self.reduced_bin_count,
                mode="nearest",
            )
            return (reconstructed + reconstructed_skip).squeeze(1)

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
            self.codebook = nn.Parameter(torch.empty(codebook_size, embedding_dim))
            _initialize_bounded_codebook(self.codebook)

        def forward(self, latents: torch.Tensor) -> TorchVectorQuantizationOutput:
            """Quantize latents and compute the stabilization loss."""
            batch_size, latent_vector_count, embedding_dim = latents.shape
            flat_latents = latents.reshape(-1, embedding_dim)

            # Compute squared Euclidean distances with the GEMM identity
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b. This avoids materializing the
            # broadcasted `[batch * M, J, d]` tensor, which reduces memory pressure
            # and lets the GPU execute the dominant work as one matrix multiply.
            flat_latents_fp32 = flat_latents.to(dtype=torch.float32)
            codebook_fp32 = self.codebook.to(dtype=torch.float32)
            latent_norms = torch.sum(flat_latents_fp32 * flat_latents_fp32, dim=1, keepdim=True)
            codebook_norms = torch.sum(codebook_fp32 * codebook_fp32, dim=1).unsqueeze(0)
            distances = (
                latent_norms
                + codebook_norms
                - 2.0 * (flat_latents_fp32 @ codebook_fp32.transpose(0, 1))
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
            self.encoder = TorchSpectralEncoder(config)
            self.vector_quantizer = TorchVectorQuantizer(
                codebook_size=config.codebook_size,
                embedding_dim=config.embedding_dim,
                commitment_weight=config.commitment_weight,
            )
            self.decoder = TorchSpectralDecoder(config)
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
            opset_version: int = 18,  # ONNX opset used by the export
        ) -> Path:
            """Export the inference-time encoder with the modern torch.export-based path.

            Purpose:
                Serialize only the sensing-node encoder boundary to ONNX using the
                recommended `torch.export`-backed exporter rather than the deprecated
                TorchScript export path.

            Args:
                output_path: Destination `.onnx` file.
                batch_size: Example batch size used during graph capture.
                opset_version: ONNX opset requested for the exported graph.

            Returns:
                The saved ONNX file path.

            Raises:
                ImportError: If `onnxscript` is missing from the active environment.
                RuntimeError: If the new exporter unexpectedly fails to return an
                    ONNX program object.
            """
            if importlib.util.find_spec("onnxscript") is None:
                raise ImportError(
                    "onnxscript is required for the torch.export-based ONNX exporter. "
                    "Install PSDCodec with the `onnx` extra or add `onnxscript` to "
                    "the active environment."
                )

            # Export from a detached CPU/eval clone so the artifact does not depend on
            # the live training device or training-mode module state.
            encoder_for_export = copy.deepcopy(self.encoder).to(device="cpu")
            encoder_for_export.eval()
            example = torch.zeros(
                batch_size,
                self.config.reduced_bin_count,
                dtype=torch.float32,
                device="cpu",
            )
            batch_dim = torch.export.Dim("batch_size")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with torch.inference_mode():
                onnx_program = torch.onnx.export(
                    encoder_for_export,
                    args=(example,),
                    f=None,
                    export_params=True,
                    opset_version=opset_version,
                    dynamo=True,
                    fallback=False,
                    input_names=["normalized_frame"],
                    output_names=["pre_quantization_latents"],
                    dynamic_shapes=({0: batch_dim},),
                )
            if onnx_program is None:
                raise RuntimeError("torch.onnx.export returned no ONNX program to serialize.")
            onnx_program.save(str(output_path))
            return output_path

else:  # pragma: no cover - exercised only when torch is unavailable

    class TorchVectorQuantizationOutput:
        """Placeholder type raised when PyTorch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Reject construction without PyTorch."""
            raise ImportError("PyTorch is required to use the torch_backend module.")

    class TorchTrainingOutput:
        """Placeholder type raised when PyTorch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Reject construction without PyTorch."""
            raise ImportError("PyTorch is required to use the torch_backend module.")

    class TorchResidualConvBlock:
        """Placeholder class raised when PyTorch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Reject construction without PyTorch."""
            raise ImportError("PyTorch is required to use the torch_backend module.")

    class TorchSpectralEncoder:
        """Placeholder class raised when PyTorch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Reject construction without PyTorch."""
            raise ImportError("PyTorch is required to use the torch_backend module.")

    class TorchSpectralDecoder:
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
