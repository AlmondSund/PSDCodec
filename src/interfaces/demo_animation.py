"""Notebook-focused animation helpers for deployment demo walkthroughs.

This module keeps plotting and frame-selection logic out of the notebook itself so
the notebook can stay as a thin orchestration layer. The helpers operate on the
stable deployment-report dataclasses exposed by :mod:`interfaces.deployment`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

from codec.exceptions import CodecConfigurationError
from interfaces.deployment import DeploymentBatchReport, DeploymentFrameReport

_plt: Any | None
_FuncAnimation: Any | None
try:
    import matplotlib.pyplot as _plt
    from matplotlib.animation import FuncAnimation as _FuncAnimation
except ImportError:  # pragma: no cover - exercised only when matplotlib is unavailable
    _plt = None
    _FuncAnimation = None


def select_animation_frame_reports(
    report: DeploymentBatchReport,
    *,
    frame_count: int = 24,
) -> tuple[DeploymentFrameReport, ...]:
    """Select animation frames that span the evaluated distortion range.

    Purpose:
        A notebook animation should show diverse examples rather than only the first
        few campaign frames. This helper samples the deployment reports across the
        sorted distortion range so the animation covers both easier and harder
        reconstructions while still preserving deterministic ordering.
    """
    if frame_count <= 0:
        raise CodecConfigurationError("frame_count must be strictly positive.")
    if not report.frame_reports:
        raise CodecConfigurationError("report.frame_reports must contain at least one frame.")

    sorted_reports = sorted(
        report.frame_reports,
        key=lambda frame_report: (
            frame_report.psd_distortion,
            frame_report.peak_frequency_error_hz,
            frame_report.frame_index,
        ),
    )
    target_size = min(frame_count, len(sorted_reports))
    quantile_positions = np.linspace(0, len(sorted_reports) - 1, num=target_size)
    selected_indices: list[int] = []
    for position in quantile_positions:
        candidate_index = int(round(float(position)))
        if candidate_index not in selected_indices:
            selected_indices.append(candidate_index)

    for candidate_index in range(len(sorted_reports)):
        if len(selected_indices) >= target_size:
            break
        if candidate_index not in selected_indices:
            selected_indices.append(candidate_index)

    return tuple(sorted_reports[candidate_index] for candidate_index in selected_indices)


def build_animation_frame_summary_rows(
    frame_reports: Sequence[DeploymentFrameReport],
) -> tuple[dict[str, float | int | str], ...]:
    """Return notebook-friendly rows summarizing the selected animation frames."""
    if not frame_reports:
        raise CodecConfigurationError("frame_reports must contain at least one frame.")
    return tuple(
        {
            "frame_index": frame_report.frame_index,
            "sequence_id": frame_report.sequence_id,
            "timestamp_utc": _format_timestamp_ms(frame_report.timestamp_ms),
            "packet_bits": frame_report.operational_bit_count,
            "psd_distortion": frame_report.psd_distortion,
            "peak_frequency_error_khz": frame_report.peak_frequency_error_hz / 1.0e3,
            "peak_power_error_db": frame_report.peak_power_error_db,
        }
        for frame_report in frame_reports
    )


def create_deployment_animation(
    frame_reports: Sequence[DeploymentFrameReport],
    *,
    interval_ms: int = 900,
    show_noise_floor: bool = True,
    plot_dbm: bool = False,
) -> Any:
    """Create a matplotlib ``FuncAnimation`` over deployment demo examples.

    Purpose:
        Animate many deployment frame reports with a stable visual layout:

        - top panel: original, preprocessing-only, and codec PSDs for the current frame,
        - bottom-left panel: packet bits versus PSD distortion for the selected frames,
        - bottom-right panel: frame-local metadata and distortion metrics.

    Args:
        frame_reports: Ordered deployment frame reports to animate.
        interval_ms: Delay between frames in milliseconds.
        show_noise_floor: Whether to plot the estimated noise floor when present.
        plot_dbm: Whether to plot PSD traces in dBm instead of linear power [mW].

    Returns:
        A ready-to-render ``matplotlib.animation.FuncAnimation`` object.
    """
    plt_module, func_animation_cls = _require_matplotlib()
    if not frame_reports:
        raise CodecConfigurationError("frame_reports must contain at least one frame.")
    if interval_ms <= 0:
        raise CodecConfigurationError("interval_ms must be strictly positive.")

    packet_bits = np.asarray(
        [frame_report.operational_bit_count for frame_report in frame_reports],
        dtype=np.float64,
    )
    psd_distortions = np.asarray(
        [frame_report.psd_distortion for frame_report in frame_reports],
        dtype=np.float64,
    )
    peak_frequency_errors_khz = np.asarray(
        [frame_report.peak_frequency_error_hz / 1.0e3 for frame_report in frame_reports],
        dtype=np.float64,
    )

    figure = plt_module.figure(figsize=(13.5, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(2.2, 1.0))
    spectrum_axis = figure.add_subplot(grid[0, :])
    summary_axis = figure.add_subplot(grid[1, 0])
    text_axis = figure.add_subplot(grid[1, 1])

    original_line = spectrum_axis.plot(
        [],
        [],
        color="#14213D",
        linewidth=2.0,
        label="Original PSD",
    )[0]
    preprocessing_line = spectrum_axis.plot(
        [],
        [],
        color="#8D99AE",
        linewidth=1.6,
        linestyle="--",
        label="Preprocessing-only",
    )[0]
    codec_line = spectrum_axis.plot(
        [],
        [],
        color="#F77F00",
        linewidth=2.0,
        label="Codec reconstruction",
    )[0]
    noise_floor_line = spectrum_axis.plot(
        [],
        [],
        color="#2A9D8F",
        linewidth=1.2,
        linestyle=":",
        label="Noise floor" if show_noise_floor else "_nolegend_",
    )[0]
    spectrum_axis.set_title("Deployment Reconstruction Walkthrough", loc="left")
    spectrum_axis.set_xlabel("Frequency [MHz]")
    spectrum_axis.set_ylabel("PSD [dBm]" if plot_dbm else "Linear power [mW]")
    spectrum_axis.grid(alpha=0.2)
    spectrum_axis.legend(loc="upper right")

    scatter = summary_axis.scatter(
        packet_bits,
        psd_distortions,
        c=peak_frequency_errors_khz,
        cmap="viridis",
        s=72,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )
    highlight = summary_axis.scatter(
        [packet_bits[0]],
        [psd_distortions[0]],
        s=220,
        facecolor="none",
        edgecolor="#D62828",
        linewidth=2.0,
    )
    summary_axis.set_title("Selected frames across the distortion range", loc="left")
    summary_axis.set_xlabel("Operational packet size [bits]")
    summary_axis.set_ylabel("PSD distortion")
    summary_axis.grid(alpha=0.2)
    colorbar = figure.colorbar(scatter, ax=summary_axis, shrink=0.85)
    colorbar.set_label("Peak-frequency error [kHz]")

    text_axis.set_axis_off()
    text_box = text_axis.text(
        0.02,
        0.98,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=10.5,
    )

    def _update(animation_index: int) -> list[Any]:
        """Update all artists for one animation frame."""
        frame_report = frame_reports[animation_index]
        frequency_mhz = frame_report.frequency_grid_hz / 1.0e6
        original_values = _power_to_plot_scale(frame_report.original_frame, plot_dbm=plot_dbm)
        preprocessing_values = _power_to_plot_scale(
            frame_report.preprocessing_only_frame,
            plot_dbm=plot_dbm,
        )
        codec_values = _power_to_plot_scale(frame_report.reconstructed_frame, plot_dbm=plot_dbm)

        original_line.set_data(frequency_mhz, original_values)
        preprocessing_line.set_data(frequency_mhz, preprocessing_values)
        codec_line.set_data(frequency_mhz, codec_values)
        if show_noise_floor and frame_report.noise_floor is not None:
            noise_floor_values = _power_to_plot_scale(frame_report.noise_floor, plot_dbm=plot_dbm)
            noise_floor_line.set_data(frequency_mhz, noise_floor_values)
            noise_floor_line.set_visible(True)
        else:
            noise_floor_values = None
            noise_floor_line.set_data([], [])
            noise_floor_line.set_visible(False)

        signal_stack = [
            original_values,
            preprocessing_values,
            codec_values,
        ]
        if noise_floor_values is not None:
            signal_stack.append(noise_floor_values)
        y_min = min(float(np.min(values)) for values in signal_stack)
        y_max = max(float(np.max(values)) for values in signal_stack)
        y_margin = max(1.0e-12, 0.08 * (y_max - y_min))
        spectrum_axis.set_xlim(float(frequency_mhz[0]), float(frequency_mhz[-1]))
        spectrum_axis.set_ylim(y_min - y_margin, y_max + y_margin)
        spectrum_axis.set_title(
            "Deployment Reconstruction Walkthrough"
            f" | frame {frame_report.frame_index}"
            f" | {frame_report.sequence_id}"
            f" | { _format_timestamp_ms(frame_report.timestamp_ms) }",
            loc="left",
        )

        highlight.set_offsets(
            np.asarray([[frame_report.operational_bit_count, frame_report.psd_distortion]])
        )
        text_box.set_text(
            "\n".join(
                (
                    f"Frame index          : {frame_report.frame_index}",
                    f"Sequence             : {frame_report.sequence_id}",
                    "Campaign / node      : "
                    f"{frame_report.campaign_label} / {frame_report.node_label}",
                    f"Timestamp (UTC)      : {_format_timestamp_ms(frame_report.timestamp_ms)}",
                    "",
                    f"Packet bits          : {frame_report.operational_bit_count}",
                    f"Rate proxy bits      : {frame_report.rate_proxy_bit_count:.2f}",
                    f"PSD distortion       : {frame_report.psd_distortion:.5f}",
                    f"Codec-only distortion: {frame_report.codec_distortion:.5f}",
                    f"Peak freq error [kHz]: {frame_report.peak_frequency_error_hz / 1.0e3:.2f}",
                    f"Peak power error [dB]: {frame_report.peak_power_error_db:.3f}",
                    f"Task distortion      : {frame_report.task_distortion:.5f}"
                    if frame_report.task_distortion is not None
                    else "Task distortion      : n/a",
                    f"Roundtrip equal      : {frame_report.roundtrip_equal}",
                )
            )
        )
        return [
            original_line,
            preprocessing_line,
            codec_line,
            noise_floor_line,
            highlight,
            text_box,
        ]

    animation = func_animation_cls(
        figure,
        _update,
        frames=len(frame_reports),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    return animation


def _require_matplotlib() -> tuple[Any, Any]:
    """Return the imported matplotlib helpers or raise a precise error."""
    if _plt is None or _FuncAnimation is None:
        raise ImportError("matplotlib is required to build the deployment animation.")
    return _plt, _FuncAnimation


def _power_to_plot_scale(
    values: np.ndarray,  # Linear-power PSD samples in milliwatts
    *,
    plot_dbm: bool,
) -> np.ndarray:
    """Convert one PSD trace into the plotting domain requested by the notebook.

    Purpose:
        The deployment pipeline stores PSD values as linear power in milliwatts so
        the distortion metrics remain physically meaningful. Notebook visualization
        can still be easier to interpret in dBm, so this helper keeps the unit
        conversion local to plotting without mutating the underlying evaluation
        reports.
    """
    plot_values = np.asarray(values, dtype=np.float64)
    if not plot_dbm:
        return plot_values

    # dBm conversion requires strictly positive power in milliwatts. Clip tiny or
    # non-positive values to a conservative floor so the plot remains finite.
    floor_power = 1.0e-20
    return np.asarray(10.0 * np.log10(np.clip(plot_values, floor_power, None)), dtype=np.float64)


def _format_timestamp_ms(
    timestamp_ms: int,  # Milliseconds since the Unix epoch
) -> str:
    """Format campaign timestamps in a notebook-friendly UTC string."""
    return datetime.fromtimestamp(timestamp_ms / 1_000.0, tz=UTC).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
