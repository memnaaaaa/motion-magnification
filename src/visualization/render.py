"""
src/visualization/render.py
---------------------------
Side-by-side video rendering and heatmap utilities.

Functions
---------
render_side_by_side(original, amplified, sep_width) -> np.ndarray
write_heatmap_overlay(frames, output_path, fps, colormap) -> None

CLI
---
python -m src.visualization.render \\
    --original  videos/wrist.avi \\
    --amplified results/videos/wrist_evm.mp4 \\
    --output    results/videos/wrist_comparison.mp4
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.io.video_io import load_video, save_video


# ---------------------------------------------------------------------------
# Core rendering function
# ---------------------------------------------------------------------------

def render_side_by_side(
    original: np.ndarray,
    amplified: np.ndarray,
    sep_width: int = 3,
) -> np.ndarray:
    """Stack original and amplified video frames side-by-side with a separator.

    If the two videos have different spatial dimensions, ``amplified`` is
    resized to match ``original``.

    Parameters
    ----------
    original : np.ndarray
        Shape (T, H, W, C), dtype float32, range [0.0, 1.0].  BGR.
    amplified : np.ndarray
        Shape (T', H', W', C), dtype float32, range [0.0, 1.0].  BGR.
        ``T'`` must equal ``T``.
    sep_width : int
        Width of the black separator bar in pixels (default 3).

    Returns
    -------
    np.ndarray
        Shape (T, H, W*2 + sep_width, C), float32.  Side-by-side frames.

    Raises
    ------
    ValueError
        If the two videos have different frame counts.
    """
    T, H, W, C = original.shape

    if amplified.shape[0] != T:
        raise ValueError(
            f"Frame count mismatch: original has {T} frames, "
            f"amplified has {amplified.shape[0]} frames."
        )

    # Resize amplified to match original dimensions if needed
    if amplified.shape[1:3] != (H, W):
        resized = np.stack(
            [
                cv2.resize(
                    amplified[t],
                    (W, H),
                    interpolation=cv2.INTER_LINEAR,
                )
                for t in range(T)
            ],
            axis=0,
        )
    else:
        resized = amplified

    separator = np.zeros((T, H, sep_width, C), dtype=np.float32)
    return np.concatenate([original, separator, resized], axis=2)


# ---------------------------------------------------------------------------
# Heatmap overlay utility
# ---------------------------------------------------------------------------

def write_heatmap_overlay(
    frames: np.ndarray,
    output_path: str,
    fps: float,
    colormap: int = cv2.COLORMAP_JET,
) -> None:
    """Write a false-colour heatmap overlay video from grayscale frames.

    Converts each frame to a normalised heatmap and saves the result.

    Parameters
    ----------
    frames : np.ndarray
        Shape (T, H, W) or (T, H, W, 1), float32.  Grayscale intensity.
    output_path : str
        Destination video file path.
    fps : float
        Output frame rate.
    colormap : int
        OpenCV colourmap constant (default ``cv2.COLORMAP_JET``).
    """
    if frames.ndim == 4:
        frames = frames[:, :, :, 0]

    T = frames.shape[0]
    heatmap_frames = np.zeros((*frames.shape[:3], 3), dtype=np.float32)

    for t in range(T):
        normalised = cv2.normalize(frames[t], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        heatmap_bgr = cv2.applyColorMap(normalised, colormap)
        heatmap_frames[t] = heatmap_bgr.astype(np.float32) / 255.0

    save_video(heatmap_frames, output_path, fps=fps)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render side-by-side original vs amplified comparison video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--original", required=True, help="Path to original video.")
    p.add_argument("--amplified", required=True, help="Path to amplified video.")
    p.add_argument("--output", required=True, help="Path to save comparison video.")
    p.add_argument("--sep-width", type=int, default=3,
                   help="Width of separator bar in pixels.")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Spatial scale factor applied when loading both videos.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    original, fps = load_video(args.original, scale=args.scale)
    amplified, _ = load_video(args.amplified, scale=args.scale)

    print(
        f"Original : {original.shape[0]} frames, "
        f"{original.shape[2]}×{original.shape[1]} px"
    )
    print(
        f"Amplified: {amplified.shape[0]} frames, "
        f"{amplified.shape[2]}×{amplified.shape[1]} px"
    )

    comparison = render_side_by_side(original, amplified, sep_width=args.sep_width)
    save_video(comparison, args.output, fps=fps)
    print(f"Saved comparison video → {args.output}")


if __name__ == "__main__":
    main()
