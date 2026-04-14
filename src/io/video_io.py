"""
src/io/video_io.py — Video I/O utilities.

Provides two pure functions:
  load_video  — read a video file → (T, H, W, C) float32 numpy array + fps
  save_video  — write a (T, H, W, C) float32 numpy array → mp4 file
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_video(
    path: str | Path,
    scale: float = 1.0,
    fallback_fps: float = 30.0,
) -> tuple[np.ndarray, float]:
    """Load a video file into a float32 numpy array.

    Args:
        path:         Path to the input video file (.mp4, .mov, etc.).
        scale:        Spatial scaling factor applied to every frame (default 1.0 = no resize).
                      Values < 1.0 downsample using cv2.INTER_AREA (good quality for shrinking).
                      Values > 1.0 upsample using cv2.INTER_LINEAR.
        fallback_fps: FPS to use when the container does not report a valid frame rate
                      (default 30.0).

    Returns:
        frames: np.ndarray, shape (T, H, W, C), dtype float32, range [0.0, 1.0].
                Channel order is BGR (OpenCV convention).
        fps:    Frame rate of the source video as a float.

    Raises:
        FileNotFoundError: If the file does not exist at ``path``.
        ValueError:        If ``scale`` is not positive, or if the video cannot be
                           opened / yields no frames.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cv2.VideoCapture could not open: {path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or fallback_fps

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            new_w = max(1, int(round(frame.shape[1] * scale)))
            new_h = max(1, int(round(frame.shape[0] * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from: {path}")

    # Stack → (T, H, W, C), convert uint8 → float32 in [0, 1]
    stacked = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return stacked, fps


def save_video(
    frames: np.ndarray,
    output_path: str | Path,
    fps: float,
    codec: str = "mp4v",
) -> None:
    """Write a float32 frame array to an mp4 video file.

    Args:
        frames:      np.ndarray, shape (T, H, W, C), dtype float32, range [0.0, 1.0].
                     Channel order must be BGR (OpenCV convention).
        output_path: Destination file path.  Parent directories are created if missing.
        fps:         Output frame rate in Hz.
        codec:       Four-character FourCC codec string (default ``"mp4v"``).
                     Try ``"avc1"`` or ``"XVID"`` if mp4v is unavailable on the platform.

    Raises:
        ValueError: If ``frames`` has an unexpected shape, dtype or range hint issues,
                    or if ``fps`` is not positive.
    """
    if frames.ndim != 4:
        raise ValueError(
            f"frames must be a 4-D array (T, H, W, C), got shape {frames.shape}"
        )
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    T, H, W, C = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    # Clip to [0, 1] then convert to uint8
    clipped = np.clip(frames, 0.0, 1.0)
    uint8_frames = (clipped * 255.0).astype(np.uint8)

    for i in range(T):
        writer.write(uint8_frames[i])

    writer.release()
