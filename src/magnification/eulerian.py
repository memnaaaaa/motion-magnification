"""
src/magnification/eulerian.py
-----------------------------
Eulerian Video Magnification (EVM) pipeline.

Decomposes each frame into a Laplacian spatial pyramid, bandpass-filters
each pyramid level along the time axis, amplifies by alpha, then adds
the result back and reconstructs.  Only the luma (Y) channel of YIQ is
processed; the I and Q chroma channels are left unchanged.

Public API
----------
run_eulerian(frames, fps, freq_low, freq_high, alpha, levels) -> np.ndarray

CLI
---
python -m src.magnification.eulerian \\
    --input  videos/wrist.avi \\
    --output results/videos/wrist_evm.mp4 \\
    --freq-low 0.5 --freq-high 2.0 --amplify 50 --levels 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from src.io.video_io import load_video, save_video
from src.pyramids.spatial import build_laplacian_pyramid, collapse_laplacian_pyramid
from src.utils.phase_utils import bgr2yiq, yiq2rgb


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _butter_bandpass(
    level_stack: np.ndarray,
    low_hz: float,
    high_hz: float,
    fps: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter along axis 0.

    Parameters
    ----------
    level_stack : np.ndarray
        Shape (T, H_l, W_l), float32.  One pyramid level across all frames.
    low_hz : float
        Lower cutoff in Hz.
    high_hz : float
        Upper cutoff in Hz.
    fps : float
        Video frame rate.
    order : int
        Butterworth filter order (default 5).

    Returns
    -------
    np.ndarray
        Filtered array, same shape as ``level_stack``, float32.
    """
    nyq = fps / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    filtered = filtfilt(b, a, level_stack.astype(np.float64), axis=0)
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_eulerian(
    frames: np.ndarray,
    fps: float,
    freq_low: float,
    freq_high: float,
    alpha: float,
    levels: int,
) -> np.ndarray:
    """Run Eulerian Video Magnification on a sequence of frames.

    Processes only the luma (Y) channel in YIQ space.  The chroma channels
    (I, Q) are left unchanged to avoid colour fringing artefacts.

    Parameters
    ----------
    frames : np.ndarray
        Shape (T, H, W, 3), dtype float32, range [0.0, 1.0].
        Channel order: BGR (OpenCV convention).
    fps : float
        Video frame rate in Hz.
    freq_low : float
        Lower temporal bandpass cutoff in Hz.
    freq_high : float
        Upper temporal bandpass cutoff in Hz (must be < fps/2).
    alpha : float
        Amplification factor (e.g., 50 for pulse; 20 for subtle vibrations).
    levels : int
        Number of Laplacian pyramid levels (typically 4–6).

    Returns
    -------
    np.ndarray
        Amplified frames, shape (T, H, W, 3), dtype float32, range [0.0, 1.0].
        Channel order: BGR.

    Raises
    ------
    ValueError
        If ``alpha <= 0``, ``levels < 1``, or cutoff frequencies are invalid.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    if freq_low <= 0:
        raise ValueError(f"freq_low must be positive, got {freq_low}")
    if freq_high <= freq_low:
        raise ValueError(f"freq_high ({freq_high}) must be > freq_low ({freq_low})")
    if freq_high >= fps / 2.0:
        raise ValueError(
            f"freq_high ({freq_high} Hz) must be < Nyquist ({fps/2.0} Hz)"
        )

    T = frames.shape[0]

    # -- Convert BGR → YIQ; process luma only ------------------------------
    yiq_frames = np.stack(
        [bgr2yiq(frames[t]) for t in range(T)], axis=0
    )  # (T, H, W, 3) float32

    luma = yiq_frames[:, :, :, 0]  # (T, H, W) float32

    # -- Build Laplacian pyramids for every frame --------------------------
    pyramids: list[list[np.ndarray]] = []
    for t in tqdm(range(T), desc="[EVM] Building pyramids", leave=False):
        pyramids.append(build_laplacian_pyramid(luma[t], levels))

    num_pyr_levels = len(pyramids[0])

    # -- Filter each pyramid level along the time axis ---------------------
    filtered_levels: list[np.ndarray] = []
    for lvl in tqdm(range(num_pyr_levels), desc="[EVM] Filtering levels", leave=False):
        # Stack level across time → (T, h_l, w_l)
        level_stack = np.stack([pyramids[t][lvl] for t in range(T)], axis=0)
        filtered_levels.append(
            _butter_bandpass(level_stack, freq_low, freq_high, fps)
        )

    # -- Amplify and add back to original pyramid, then collapse -----------
    result_luma = np.empty_like(luma)
    for t in tqdm(range(T), desc="[EVM] Reconstructing", leave=False):
        amplified_pyr = [
            pyramids[t][lvl] + (alpha * filtered_levels[lvl][t] if lvl >= 2 else 0.0)
            for lvl in range(num_pyr_levels)
        ]
        result_luma[t] = collapse_laplacian_pyramid(amplified_pyr)

    result_luma = np.clip(result_luma, 0.0, 1.0)

    # -- Replace luma and convert YIQ → BGR --------------------------------
    result_yiq = yiq_frames.copy()
    result_yiq[:, :, :, 0] = result_luma

    result_bgr = np.stack(
        [
            cv2.cvtColor(yiq2rgb(result_yiq[t]), cv2.COLOR_RGB2BGR)
            for t in range(T)
        ],
        axis=0,
    )  # (T, H, W, 3) float32 in [0, 1]

    return result_bgr


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Eulerian Video Magnification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to input video file.")
    p.add_argument("--output", required=True, help="Path to save amplified video.")
    p.add_argument(
        "--freq-low", type=float, required=True,
        help="Lower temporal bandpass cutoff in Hz.",
    )
    p.add_argument(
        "--freq-high", type=float, required=True,
        help="Upper temporal bandpass cutoff in Hz.",
    )
    p.add_argument(
        "--amplify", type=float, required=True,
        help="Amplification factor alpha.",
    )
    p.add_argument(
        "--levels", type=int, default=4,
        help="Number of Laplacian pyramid levels.",
    )
    p.add_argument(
        "--scale", type=float, default=1.0,
        help="Spatial scale factor applied to frames before processing.",
    )
    p.add_argument(
        "--fps", type=float, default=0.0,
        help="Override detected FPS (0 = auto-detect from video).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    frames, detected_fps = load_video(args.input, scale=args.scale)
    fps = args.fps if args.fps > 0 else detected_fps

    print(
        f"Loaded {frames.shape[0]} frames at {fps:.2f} fps  "
        f"({frames.shape[2]}×{frames.shape[1]} px)"
    )

    result = run_eulerian(
        frames=frames,
        fps=fps,
        freq_low=args.freq_low,
        freq_high=args.freq_high,
        alpha=args.amplify,
        levels=args.levels,
    )

    save_video(result, args.output, fps=fps)
    print(f"Saved amplified video → {args.output}")


if __name__ == "__main__":
    main()
