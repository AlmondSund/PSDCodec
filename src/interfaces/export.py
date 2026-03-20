"""Deployment export helpers aligned with the encoder-only ONNX boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast


def export_encoder_to_onnx(
    model: Any,  # PyTorch learned codec exposing `export_encoder_to_onnx`
    output_path: str | Path,  # Destination `.onnx` path
    *,
    batch_size: int = 1,  # Example batch size for tracing
    opset_version: int = 18,  # ONNX opset requested by the export
) -> Path:
    """Export the encoder-only deployment boundary to ONNX."""
    destination = Path(output_path)
    export_method = getattr(model, "export_encoder_to_onnx", None)
    if export_method is None:
        raise TypeError("model must expose an export_encoder_to_onnx(output_path, ...) method.")
    return cast(
        Path,
        export_method(destination, batch_size=batch_size, opset_version=opset_version),
    )
