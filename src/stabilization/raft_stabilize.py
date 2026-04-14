"""
src/stabilization/raft_stabilize.py
-------------------------------------
Video stabilization using RAFT optical flow.

Estimates camera motion between consecutive frames with RAFT
(Recurrent All-Pairs Field Transforms, Teed & Deng 2020), builds a
smooth camera trajectory, and warps each frame to compensate for shake.
The output is a stabilized video suitable for motion magnification.

Algorithm
---------
1. Load pretrained RAFT model from torchvision.
2. For each pair of consecutive frames (t → t+1), compute dense optical flow.
3. Estimate global translation (tx, ty) as the median of the flow field —
   median is robust to moving objects and occlusions.
4. Accumulate translations to build the raw camera trajectory.
5. Smooth the trajectory with a Gaussian filter (sigma = smoothing_radius).
6. Compute per-frame correction = smooth_trajectory − raw_trajectory.
7. Warp each frame by its correction using cv2.warpAffine.
8. Optionally crop the black borders introduced by warping.

Public API
----------
stabilize_video(frames, model_size, smoothing_radius, crop_borders, device)
    -> np.ndarray

CLI
---
python -m src.stabilization.raft_stabilize \\
    --input  videos/raw.mp4 \\
    --output videos/stabilized.mp4 \\
    --model  small \\
    --smoothing-radius 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from src.io.video_io import load_video, save_video


# ---------------------------------------------------------------------------
# RAFT model loader
# ---------------------------------------------------------------------------

def _load_raft(model_size: str, device: str) -> tuple:
    """Load a pretrained RAFT optical-flow model and its preprocessing transforms.

    Parameters
    ----------
    model_size : str
        ``"small"`` (faster, ~21 MB) or ``"large"`` (more accurate, ~90 MB).
        Weights are downloaded to the torch hub cache on first call.
    device : str
        PyTorch device string.

    Returns
    -------
    model : torch.nn.Module
        RAFT model in eval mode, moved to ``device``.
    transforms : callable
        Preprocessing transform that converts uint8 RGB (N, 3, H, W) tensors
        to the float range expected by the model.
    """
    from torchvision.models.optical_flow import (
        raft_large, raft_small,
        Raft_Large_Weights, Raft_Small_Weights,
    )

    if model_size == "small":
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=True)
    elif model_size == "large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True)
    else:
        raise ValueError(f"model_size must be 'small' or 'large', got '{model_size}'")

    model = model.to(device).eval()
    return model, weights.transforms()


# ---------------------------------------------------------------------------
# Frame preprocessing / postprocessing helpers
# ---------------------------------------------------------------------------

def _frame_to_raft_tensor(
    frame_bgr_f32: np.ndarray,
    device: str,
) -> torch.Tensor:
    """Convert a float32 BGR frame [0,1] to a uint8 RGB tensor (1, 3, H, W).

    RAFT (torchvision) expects uint8 RGB input; the model's transforms
    normalize it internally.

    Parameters
    ----------
    frame_bgr_f32 : np.ndarray
        Shape (H, W, 3), float32, range [0.0, 1.0], BGR channel order.
    device : str
        Target device.

    Returns
    -------
    torch.Tensor
        Shape (1, 3, H, W), dtype uint8, on ``device``.
    """
    uint8_bgr = (np.clip(frame_bgr_f32, 0.0, 1.0) * 255).astype(np.uint8)
    uint8_rgb = cv2.cvtColor(uint8_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(uint8_rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple]:
    """Pad spatial dimensions so H and W are divisible by ``multiple``.

    RAFT internally uses downsampling operations that require H and W to be
    divisible by 8.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape (N, C, H, W).
    multiple : int
        Required divisor (default 8).

    Returns
    -------
    padded : torch.Tensor
        Shape (N, C, H', W') where H' and W' are multiples of ``multiple``.
    padding : tuple[int, int, int, int]
        ``(pad_left, pad_right, pad_top, pad_bottom)`` values applied.
    """
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    padding = (0, pad_w, 0, pad_h)  # F.pad uses (left, right, top, bottom)
    padded = F.pad(tensor, padding, mode="replicate")
    return padded, padding


# ---------------------------------------------------------------------------
# Optical flow computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_flow_translation(
    frame1: np.ndarray,
    frame2: np.ndarray,
    model: "torch.nn.Module",
    transforms: callable,
    device: str,
) -> tuple[float, float]:
    """Estimate the global translation between two frames using RAFT.

    Parameters
    ----------
    frame1, frame2 : np.ndarray
        Consecutive frames, shape (H, W, 3), float32 BGR [0,1].
    model : torch.nn.Module
        RAFT model in eval mode.
    transforms : callable
        RAFT preprocessing transform.
    device : str
        PyTorch device.

    Returns
    -------
    tx, ty : float
        Median horizontal and vertical displacement from frame1 to frame2.
    """
    t1 = _frame_to_raft_tensor(frame1, device)
    t2 = _frame_to_raft_tensor(frame2, device)

    t1_padded, padding = _pad_to_multiple(t1)
    t2_padded, _ = _pad_to_multiple(t2)

    # Apply RAFT preprocessing transforms
    t1_pre, t2_pre = transforms(t1_padded, t2_padded)

    # RAFT returns a list of flow predictions; the last is the most refined
    flow_predictions = model(t1_pre, t2_pre)
    flow = flow_predictions[-1]  # (1, 2, H_padded, W_padded)

    # Crop back to original size (remove padding)
    _, _, H_pad, W_pad = flow.shape
    H_orig = H_pad - padding[3]
    W_orig = W_pad - padding[1]
    flow = flow[:, :, :H_orig, :W_orig]

    flow_np = flow[0].cpu().numpy()  # (2, H, W)
    tx = float(np.median(flow_np[0]))  # horizontal
    ty = float(np.median(flow_np[1]))  # vertical

    return tx, ty


# ---------------------------------------------------------------------------
# Main stabilization pipeline
# ---------------------------------------------------------------------------

def stabilize_video(
    frames: np.ndarray,
    model_size: str = "small",
    smoothing_radius: float = 30.0,
    crop_borders: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """Stabilize a video using RAFT optical-flow-based trajectory smoothing.

    Parameters
    ----------
    frames : np.ndarray
        Shape (T, H, W, 3), dtype float32, range [0.0, 1.0].  BGR channel order.
    model_size : str
        ``"small"`` or ``"large"`` (default ``"small"``).
    smoothing_radius : float
        Standard deviation (in frames) for Gaussian trajectory smoothing.
        Larger values remove more camera shake but may introduce lag artefacts
        (default 30.0).
    crop_borders : bool
        If True (default), crop the black borders introduced by warping.
        The output will be slightly smaller than the input.
    device : str or None
        PyTorch device.  Auto-detects CUDA if None.

    Returns
    -------
    np.ndarray
        Stabilized frames.  Shape (T, H', W', 3) if ``crop_borders=True``,
        or (T, H, W, 3) if False.  dtype float32, range [0.0, 1.0], BGR.

    Raises
    ------
    ValueError
        If ``frames`` is not float32 or has wrong ndim, or ``model_size``
        is invalid.
    """
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            f"frames must be shape (T, H, W, 3), got {frames.shape}"
        )
    if frames.dtype != np.float32:
        raise ValueError(f"frames must be float32, got {frames.dtype}")
    if model_size not in ("small", "large"):
        raise ValueError(f"model_size must be 'small' or 'large', got '{model_size}'")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Stabilize] Device: {device}  |  RAFT model: {model_size}")

    T, H, W, _ = frames.shape

    # Load RAFT
    model, transforms = _load_raft(model_size, device)

    # ---------------------------------------------------------------
    # Step 1: Estimate per-frame translations (consecutive pairs)
    # ---------------------------------------------------------------
    # translations[t] = (tx, ty) displacement from frame t to frame t+1
    translations = np.zeros((T - 1, 2), dtype=np.float64)

    for t in tqdm(range(T - 1), desc="[Stabilize] Computing optical flow"):
        tx, ty = _compute_flow_translation(
            frames[t], frames[t + 1], model, transforms, device
        )
        translations[t, 0] = tx
        translations[t, 1] = ty

    # ---------------------------------------------------------------
    # Step 2: Build and smooth the camera trajectory
    # ---------------------------------------------------------------
    # Cumulative sum gives the position of the camera at each frame
    # relative to frame 0
    trajectory = np.zeros((T, 2), dtype=np.float64)
    trajectory[1:] = np.cumsum(translations, axis=0)

    # Smooth trajectory
    smooth = np.zeros_like(trajectory)
    smooth[:, 0] = gaussian_filter1d(trajectory[:, 0], sigma=smoothing_radius)
    smooth[:, 1] = gaussian_filter1d(trajectory[:, 1], sigma=smoothing_radius)

    # Correction: how much to shift each frame
    correction = smooth - trajectory  # (T, 2)

    # ---------------------------------------------------------------
    # Step 3: Warp frames by correction transform
    # ---------------------------------------------------------------
    stabilized = np.empty_like(frames)
    for t in range(T):
        dx, dy = correction[t, 0], correction[t, 1]
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        stabilized[t] = cv2.warpAffine(
            frames[t], M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    # ---------------------------------------------------------------
    # Step 4: Crop borders (optional)
    # ---------------------------------------------------------------
    if crop_borders:
        max_dx = int(np.ceil(np.max(np.abs(correction[:, 0])))) + 1
        max_dy = int(np.ceil(np.max(np.abs(correction[:, 1])))) + 1

        x0 = max_dx
        x1 = W - max_dx
        y0 = max_dy
        y1 = H - max_dy

        if x1 <= x0 or y1 <= y0:
            print(
                "[Stabilize] WARNING: Correction too large to crop safely. "
                "Returning uncropped stabilized video."
            )
        else:
            stabilized = stabilized[:, y0:y1, x0:x1, :]
            print(
                f"[Stabilize] Cropped to {stabilized.shape[2]}×{stabilized.shape[1]} px "
                f"(removed {max_dx}px H / {max_dy}px V margins)"
            )

    return stabilized.astype(np.float32)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stabilize a video using RAFT optical flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to input (shaky) video.")
    p.add_argument("--output", required=True, help="Path to save stabilized video.")
    p.add_argument(
        "--model", default="small", choices=["small", "large"],
        help="RAFT model size ('small' is faster; 'large' is more accurate).",
    )
    p.add_argument(
        "--smoothing-radius", type=float, default=30.0,
        help="Gaussian σ (frames) for trajectory smoothing.",
    )
    p.add_argument(
        "--no-crop", action="store_true",
        help="Disable border cropping (output keeps original frame size).",
    )
    p.add_argument(
        "--scale", type=float, default=1.0,
        help="Spatial scale factor applied when loading the video.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    frames, fps = load_video(args.input, scale=args.scale)
    print(
        f"Loaded {frames.shape[0]} frames at {fps:.2f} fps  "
        f"({frames.shape[2]}×{frames.shape[1]} px)"
    )

    stabilized = stabilize_video(
        frames=frames,
        model_size=args.model,
        smoothing_radius=args.smoothing_radius,
        crop_borders=not args.no_crop,
    )

    save_video(stabilized, args.output, fps=fps)
    print(f"Saved stabilized video → {args.output}")


if __name__ == "__main__":
    main()
