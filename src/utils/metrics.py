"""
src/utils/metrics.py
--------------------
Temporal SNR computation and parameter-sweep utilities.

Functions
---------
compute_temporal_snr(frames, fps, freq_low, freq_high, signal_roi, noise_roi)
    -> float

snr_alpha_sweep(frames, fps, freq_low, freq_high, alphas, levels,
                signal_roi, noise_roi, output_path) -> dict[float, float]

CLI
---
python -m src.utils.metrics \\
    --input  videos/wrist.avi \\
    --alphas 10 20 30 50 75 100 \\
    --freq-low 0.5 --freq-high 2.0 \\
    --signal-roi 100 50 200 150 \\
    --noise-roi  10  10  80  80 \\
    --output results/plots/snr_sweep_wrist.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from src.io.video_io import load_video
from src.magnification.eulerian import run_eulerian
from src.utils.phase_utils import bgr2yiq


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _butter_luma(
    frames: np.ndarray,
    freq_low: float,
    freq_high: float,
    fps: float,
    order: int = 5,
) -> np.ndarray:
    """Bandpass-filter the luma channel of a video along the time axis.

    Parameters
    ----------
    frames : np.ndarray
        Shape (T, H, W, 3), float32, BGR.
    freq_low, freq_high : float
        Bandpass bounds in Hz.
    fps : float
        Frame rate.
    order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Shape (T, H, W), float32.  Filtered luma signal.
    """
    T = frames.shape[0]
    luma = np.stack([bgr2yiq(frames[t])[:, :, 0] for t in range(T)], axis=0)  # (T, H, W)

    nyq = fps / 2.0
    b, a = butter(order, [freq_low / nyq, freq_high / nyq], btype="band")
    return filtfilt(b, a, luma.astype(np.float64), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_temporal_snr(
    frames: np.ndarray,
    fps: float,
    freq_low: float,
    freq_high: float,
    signal_roi: tuple[int, int, int, int],
    noise_roi: tuple[int, int, int, int],
) -> float:
    """Compute the temporal SNR (dB) of a processed video.

    SNR is defined as::

        signal_power = mean spatial variance of bandpass-filtered luma in signal_roi
        noise_power  = mean spatial variance of bandpass-filtered luma in noise_roi
        SNR_dB       = 10 * log10(signal_power / noise_power)

    Parameters
    ----------
    frames : np.ndarray
        Shape (T, H, W, 3), float32, BGR.
    fps : float
        Video frame rate in Hz.
    freq_low : float
        Lower bandpass cutoff in Hz.
    freq_high : float
        Upper bandpass cutoff in Hz.
    signal_roi : tuple[int, int, int, int]
        Bounding box ``(x0, y0, x1, y1)`` of the signal region (e.g., neck/wrist).
    noise_roi : tuple[int, int, int, int]
        Bounding box ``(x0, y0, x1, y1)`` of a static background region.

    Returns
    -------
    float
        SNR in decibels.  Returns ``float('nan')`` if noise power is zero.

    Raises
    ------
    ValueError
        If ROI coordinates are out of range or frequencies are invalid.
    """
    if freq_low <= 0 or freq_high <= freq_low or freq_high >= fps / 2.0:
        raise ValueError(
            f"Invalid frequency bounds: freq_low={freq_low}, freq_high={freq_high}, "
            f"Nyquist={fps/2.0}"
        )

    sx0, sy0, sx1, sy1 = signal_roi
    nx0, ny0, nx1, ny1 = noise_roi
    H, W = frames.shape[1:3]

    for label, x0, y0, x1, y1 in [
        ("signal_roi", sx0, sy0, sx1, sy1),
        ("noise_roi", nx0, ny0, nx1, ny1),
    ]:
        if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
            raise ValueError(
                f"{label} ({x0},{y0},{x1},{y1}) is out of bounds for frame size {H}×{W}."
            )

    filtered_luma = _butter_luma(frames, freq_low, freq_high, fps)  # (T, H, W)

    # Mean temporal variance across pixels in each ROI
    sig_pixels = filtered_luma[:, sy0:sy1, sx0:sx1]  # (T, roi_H, roi_W)
    noise_pixels = filtered_luma[:, ny0:ny1, nx0:nx1]

    signal_power = float(np.var(sig_pixels, axis=0).mean())
    noise_power = float(np.var(noise_pixels, axis=0).mean())

    if noise_power == 0.0:
        return float("nan")

    return 10.0 * np.log10(signal_power / noise_power)


def snr_alpha_sweep(
    frames: np.ndarray,
    fps: float,
    freq_low: float,
    freq_high: float,
    alphas: list[float],
    levels: int,
    signal_roi: tuple[int, int, int, int],
    noise_roi: tuple[int, int, int, int],
    output_path: str | Path | None = None,
) -> dict[float, float]:
    """Sweep amplification factor alpha and compute temporal SNR at each value.

    For each alpha in ``alphas``:
    1. Run the Eulerian pipeline on ``frames``.
    2. Compute temporal SNR of the output using ``compute_temporal_snr``.

    Parameters
    ----------
    frames : np.ndarray
        Shape (T, H, W, 3), float32, BGR input video.
    fps : float
        Video frame rate.
    freq_low, freq_high : float
        Temporal bandpass bounds in Hz.
    alphas : list[float]
        Amplification factors to evaluate.
    levels : int
        Laplacian pyramid levels for the Eulerian pipeline.
    signal_roi : tuple[int, int, int, int]
        Signal region bounding box ``(x0, y0, x1, y1)``.
    noise_roi : tuple[int, int, int, int]
        Background noise region bounding box ``(x0, y0, x1, y1)``.
    output_path : str or Path or None
        If given, save the SNR-vs-alpha plot to this path (.png).

    Returns
    -------
    dict[float, float]
        Mapping ``alpha → SNR_dB`` for each evaluated alpha.
    """
    results: dict[float, float] = {}

    for alpha in tqdm(alphas, desc="[SNR] Alpha sweep"):
        amplified = run_eulerian(
            frames=frames,
            fps=fps,
            freq_low=freq_low,
            freq_high=freq_high,
            alpha=alpha,
            levels=levels,
        )
        snr = compute_temporal_snr(
            frames=amplified,
            fps=fps,
            freq_low=freq_low,
            freq_high=freq_high,
            signal_roi=signal_roi,
            noise_roi=noise_roi,
        )
        results[alpha] = snr
        print(f"  alpha={alpha:>6.1f}  SNR={snr:+.2f} dB")

    if output_path is not None:
        _plot_snr_sweep(results, output_path, freq_low, freq_high)

    return results


def _plot_snr_sweep(
    results: dict[float, float],
    output_path: str | Path,
    freq_low: float,
    freq_high: float,
) -> None:
    """Save an SNR-vs-alpha plot to ``output_path``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    alphas = sorted(results.keys())
    snrs = [results[a] for a in alphas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, snrs, marker="o", linewidth=2, color="steelblue")
    ax.set_xlabel("Amplification factor α", fontsize=13)
    ax.set_ylabel("Temporal SNR (dB)", fontsize=13)
    ax.set_title(
        f"SNR vs α  |  bandpass [{freq_low}–{freq_high}] Hz", fontsize=14
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"SNR plot saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute temporal SNR vs amplification factor sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to input video.")
    p.add_argument("--alphas", nargs="+", type=float, required=True,
                   help="Amplification factor values to sweep (space-separated).")
    p.add_argument("--freq-low", type=float, required=True,
                   help="Lower temporal bandpass cutoff in Hz.")
    p.add_argument("--freq-high", type=float, required=True,
                   help="Upper temporal bandpass cutoff in Hz.")
    p.add_argument("--levels", type=int, default=4,
                   help="Laplacian pyramid levels for Eulerian pipeline.")
    p.add_argument("--signal-roi", nargs=4, type=int, required=True,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="Signal region bounding box in pixels.")
    p.add_argument("--noise-roi", nargs=4, type=int, required=True,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="Background noise region bounding box in pixels.")
    p.add_argument("--output", required=True,
                   help="Path to save SNR-vs-alpha plot (.png).")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Spatial scale factor when loading the video.")
    p.add_argument("--fps", type=float, default=0.0,
                   help="Override detected FPS (0 = auto-detect).")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    frames, detected_fps = load_video(args.input, scale=args.scale)
    fps = args.fps if args.fps > 0 else detected_fps

    signal_roi = tuple(args.signal_roi)
    noise_roi = tuple(args.noise_roi)

    print(f"Loaded {frames.shape[0]} frames at {fps:.2f} fps")
    print(f"Signal ROI: {signal_roi}")
    print(f"Noise  ROI: {noise_roi}")

    snr_alpha_sweep(
        frames=frames,
        fps=fps,
        freq_low=args.freq_low,
        freq_high=args.freq_high,
        alphas=args.alphas,
        levels=args.levels,
        signal_roi=signal_roi,
        noise_roi=noise_roi,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
