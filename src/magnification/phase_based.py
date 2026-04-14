"""
src/magnification/phase_based.py
---------------------------------
Phase-Based Motion Magnification pipeline.

CLI
---
python -m src.magnification.phase_based \\
    --input  videos/wrist.avi \\
    --output results/videos/wrist_phase.mp4 \\
    --freq-low 0.5 --freq-high 2.0 --amplify 25 \\
    --colorspace luma3 --pyramid half_octave --sigma 2.0 --batch 4 --scale 0.5
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.io.video_io import load_video, save_video
from src.pyramids.steerable import SteerablePyramid, SuboctaveSP
from src.pyramids.pyramid_utils import build_level, build_level_batch, recon_level_batch
from src.utils.phase_utils import (
    bgr2yiq,
    yiq2rgb,
    get_fft2_batch,
    bandpass_filter,
)

_EPS = 1e-6
_PYRAMID_TYPES = ("full_octave", "half_octave", "smooth_half_octave", "smooth_quarter_octave")
_COLORSPACES = ("luma1", "luma3", "gray", "yiq", "rgb")


# ---------------------------------------------------------------------------
# Phase-based processor (adapted from phase_based_processing.py)
# ---------------------------------------------------------------------------

class _PhaseBased:
    """Phase-based motion magnification processor (internal class).

    Parameters
    ----------
    sigma : float
        Std-dev of the Gaussian kernel for amplitude-weighted phase blurring.
        Set to 0 to disable blurring.
    transfer_function : torch.Tensor
        Shape (1, T, 1, 1), complex64.  Frequency-domain bandpass filter.
    phase_mag : float
        Phase amplification factor (alpha).
    attenuate : bool
        If True, attenuate frequencies outside the passband.
    ref_idx : int
        Index of the reference (DC) frame for phase difference computation.
    batch_size : int
        Number of filters processed simultaneously (GPU memory trade-off).
    device : str
        PyTorch device string (``"cuda"`` or ``"cpu"``).
    eps : float
        Small value to prevent division by zero (default 1e-6).
    """

    def __init__(
        self,
        sigma: float,
        transfer_function: torch.Tensor,
        phase_mag: float,
        attenuate: bool,
        ref_idx: int,
        batch_size: int,
        device: str,
        eps: float = _EPS,
    ) -> None:
        self.sigma = sigma
        self.transfer_function = transfer_function
        self.phase_mag = phase_mag
        self.attenuate = attenuate
        self.ref_idx = ref_idx
        self.batch_size = batch_size
        self.device = device
        self.eps = eps
        self.gauss_kernel = self._make_gauss_kernel()

    def _make_gauss_kernel(self) -> torch.Tensor:
        """Build a 2-D Gaussian convolution kernel for amplitude-weighted blurring."""
        import math
        ksize = max(3, int(math.ceil(4 * self.sigma)) - 1)
        if ksize % 2 == 0:
            ksize += 1
        gk = cv2.getGaussianKernel(ksize=ksize, sigma=self.sigma)
        kernel_2d = (gk @ gk.T).astype(np.float32)
        return (
            torch.tensor(kernel_2d)
            .to(self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def process_single_channel(
        self,
        frames_tensor: torch.Tensor,
        filters_tensor: torch.Tensor,
        video_dft: torch.Tensor,
    ) -> torch.Tensor:
        """Apply phase-based magnification to a single-channel video.

        Parameters
        ----------
        frames_tensor : torch.Tensor
            Shape (T, H, W), float32.  Source frames (used only for shape info).
        filters_tensor : torch.Tensor
            Shape (num_filters, H, W), float32.  Steerable pyramid filters.
        video_dft : torch.Tensor
            Shape (T, H, W), complex64.  Shifted 2-D DFT of every frame.

        Returns
        -------
        torch.Tensor
            Shape (T, H, W), float32.  Reconstructed magnified frames.
        """
        num_frames, _, _ = frames_tensor.shape
        num_filters, h, w = filters_tensor.shape

        recon_dft = torch.zeros((num_frames, h, w), dtype=torch.complex64).to(self.device)
        phase_deltas = torch.zeros(
            (self.batch_size, num_frames, h, w), dtype=torch.float32
        ).to(self.device)

        total_levels = (num_filters - 2 + self.batch_size - 1) // self.batch_size
        current_level = 0

        for level in tqdm(
            range(1, num_filters - 1, self.batch_size),
            desc="[Phase] Processing levels",
            total=total_levels,
            leave=False,
        ):
            current_level += 1

            # Current filter batch (may be smaller than batch_size at the end)
            filter_batch = filters_tensor[level: level + self.batch_size]
            actual_batch = filter_batch.shape[0]

            # Reference frame phase
            ref_pyr = build_level_batch(
                video_dft[self.ref_idx].unsqueeze(0), filter_batch
            )
            ref_phase = torch.angle(ref_pyr)

            # Phase deltas for every frame
            for vid_idx in range(num_frames):
                curr_pyr = build_level_batch(
                    video_dft[vid_idx].unsqueeze(0), filter_batch
                )
                _delta = torch.angle(curr_pyr) - ref_phase
                # Wrap to [-π, π]
                phase_deltas[:actual_batch, vid_idx] = (
                    (torch.pi + _delta) % (2 * torch.pi)
                ) - torch.pi

            # Temporal bandpass filter (frequency domain)
            pd_slice = phase_deltas[:actual_batch]
            pd_slice = torch.fft.ifft(
                self.transfer_function * torch.fft.fft(pd_slice, dim=1),
                dim=1,
            ).real
            phase_deltas[:actual_batch] = pd_slice

            # Amplify and reconstruct
            for vid_idx in range(num_frames):
                vid_dft = video_dft[vid_idx].unsqueeze(0)
                curr_pyr = build_level_batch(vid_dft, filter_batch)
                delta = phase_deltas[:actual_batch, vid_idx].unsqueeze(1)

                # Amplitude-weighted blurring
                if self.sigma != 0:
                    amplitude_weight = (torch.abs(curr_pyr) + self.eps).unsqueeze(1)
                    weight = F.conv2d(
                        amplitude_weight, self.gauss_kernel, padding="same"
                    ).squeeze(1)
                    delta = F.conv2d(
                        amplitude_weight * delta, self.gauss_kernel, padding="same"
                    ).squeeze(1)
                    delta = delta / weight

                modified_phase = delta * self.phase_mag

                if self.attenuate:
                    curr_pyr = torch.abs(curr_pyr) * (ref_pyr / torch.abs(ref_pyr))

                curr_pyr = curr_pyr * torch.exp(1.0j * modified_phase)

                recon_dft[vid_idx] += recon_level_batch(curr_pyr, filter_batch).sum(dim=0)

        # Add back unchanged low-pass component
        lopass = filters_tensor[-1]
        for vid_idx in range(num_frames):
            pyr_lo = build_level(video_dft[vid_idx], lopass)
            dft_lo = torch.fft.fftshift(torch.fft.fft2(pyr_lo))
            recon_dft[vid_idx] += dft_lo * lopass

        return torch.fft.ifft2(
            torch.fft.ifftshift(recon_dft, dim=(1, 2)), dim=(1, 2)
        ).real


# ---------------------------------------------------------------------------
# Pyramid factory
# ---------------------------------------------------------------------------

def _make_pyramid(pyramid_type: str, depth: int) -> SteerablePyramid | SuboctaveSP:
    """Construct the requested steerable pyramid object.

    Parameters
    ----------
    pyramid_type : str
        One of ``"full_octave"``, ``"half_octave"``, ``"smooth_half_octave"``,
        ``"smooth_quarter_octave"``.
    depth : int
        Pyramid depth (derived from frame dimensions).

    Returns
    -------
    SteerablePyramid or SuboctaveSP
    """
    if pyramid_type == "full_octave":
        return SteerablePyramid(depth=depth, orientations=4, filters_per_octave=1,
                                twidth=1.0, complex_pyr=True)
    elif pyramid_type == "half_octave":
        return SteerablePyramid(depth=depth, orientations=8, filters_per_octave=2,
                                twidth=0.75, complex_pyr=True)
    elif pyramid_type == "smooth_half_octave":
        return SuboctaveSP(depth=depth, orientations=8, filters_per_octave=2,
                           cos_order=6, complex_pyr=True)
    elif pyramid_type == "smooth_quarter_octave":
        return SuboctaveSP(depth=depth, orientations=8, filters_per_octave=4,
                           cos_order=6, complex_pyr=True)
    else:
        raise ValueError(
            f"Unknown pyramid_type '{pyramid_type}'. "
            f"Choose from: {_PYRAMID_TYPES}"
        )


# ---------------------------------------------------------------------------
# Colorspace helpers
# ---------------------------------------------------------------------------

def _to_processing_space(
    frames: np.ndarray,
    colorspace: str,
) -> np.ndarray:
    """Convert BGR float32 frames to the processing colorspace.

    Parameters
    ----------
    frames : np.ndarray
        Shape (T, H, W, 3), float32, BGR.
    colorspace : str
        Target colorspace name.

    Returns
    -------
    np.ndarray
        Converted frames.  Shape is (T, H, W, 3) for multi-channel modes
        and (T, H, W) for single-channel modes (luma1, gray).
    """
    T = frames.shape[0]
    if colorspace in ("luma3", "yiq"):
        return np.stack([bgr2yiq(frames[t]) for t in range(T)], axis=0)
    elif colorspace == "luma1":
        return np.stack([bgr2yiq(frames[t])[:, :, 0] for t in range(T)], axis=0)
    elif colorspace == "gray":
        return np.stack([cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY) for t in range(T)], axis=0)
    elif colorspace == "rgb":
        return np.stack([cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB) for t in range(T)], axis=0)
    else:
        raise ValueError(f"Unknown colorspace '{colorspace}'. Choose from: {_COLORSPACES}")


def _to_bgr(
    result: np.ndarray,
    original: np.ndarray,
    colorspace: str,
) -> np.ndarray:
    """Convert processed frames back to BGR float32 [0, 1].

    Parameters
    ----------
    result : torch.Tensor or np.ndarray
        Processed output in the processing colorspace.
        Shape (T, H, W, 3) or (T, H, W) depending on colorspace.
    original : np.ndarray
        Original frames in the processing colorspace (T, H, W, 3) or (T, H, W).
        Used for luma3 to restore IQ channels.
    colorspace : str
        Processing colorspace used.

    Returns
    -------
    np.ndarray
        Shape (T, H, W, 3), float32, BGR, [0.0, 1.0].
    """
    T = result.shape[0]

    if colorspace == "luma3":
        # Replace Y channel; keep IQ unchanged
        yiq_result = original.copy()
        yiq_result[:, :, :, 0] = result
        return np.stack(
            [cv2.cvtColor(yiq2rgb(yiq_result[t]), cv2.COLOR_RGB2BGR) for t in range(T)],
            axis=0,
        )
    elif colorspace == "yiq":
        return np.stack(
            [cv2.cvtColor(yiq2rgb(result[t]), cv2.COLOR_RGB2BGR) for t in range(T)],
            axis=0,
        )
    elif colorspace == "luma1":
        # Stretch single-channel back to BGR grayscale
        out = np.stack(
            [cv2.cvtColor(
                cv2.normalize(result[t], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
                cv2.COLOR_GRAY2BGR,
            ).astype(np.float32) / 255.0
            for t in range(T)],
            axis=0,
        )
        return out
    elif colorspace == "gray":
        return np.stack(
            [cv2.cvtColor(
                cv2.normalize(result[t], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
                cv2.COLOR_GRAY2BGR,
            ).astype(np.float32) / 255.0
            for t in range(T)],
            axis=0,
        )
    elif colorspace == "rgb":
        return np.stack(
            [np.clip(cv2.cvtColor(result[t], cv2.COLOR_RGB2BGR), 0.0, 1.0)
             for t in range(T)],
            axis=0,
        )
    else:
        raise ValueError(f"Unknown colorspace '{colorspace}'.")


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_phase_based(
    frames: np.ndarray,
    fps: float,
    freq_low: float,
    freq_high: float,
    alpha: float,
    *,
    colorspace: str = "luma3",
    pyramid_type: str = "half_octave",
    sigma: float = 2.0,
    attenuate: bool = False,
    ref_idx: int = 0,
    batch_size: int = 4,
    depth: int | None = None,
) -> np.ndarray:
    """Run phase-based motion magnification on a sequence of frames.

    Parameters
    ----------
    frames : np.ndarray
        Shape (T, H, W, 3), dtype float32, range [0.0, 1.0].  BGR channel order.
    fps : float
        Video frame rate in Hz.
    freq_low : float
        Lower temporal bandpass cutoff in Hz.
    freq_high : float
        Upper temporal bandpass cutoff in Hz (must be < fps/2).
    alpha : float
        Phase amplification factor.
    colorspace : str
        Processing colorspace.  One of ``"luma1"``, ``"luma3"``, ``"gray"``,
        ``"yiq"``, ``"rgb"`` (default ``"luma3"``).
    pyramid_type : str
        Complex steerable pyramid variant.  One of ``"full_octave"``,
        ``"half_octave"``, ``"smooth_half_octave"``, ``"smooth_quarter_octave"``
        (default ``"half_octave"``).
    sigma : float
        Std-dev of amplitude-weighted Gaussian blurring (0 = disabled).
    attenuate : bool
        Attenuate frequencies outside the passband if True.
    ref_idx : int
        Index of the reference (DC) frame (default 0).
    batch_size : int
        Number of filters processed simultaneously per GPU pass (default 4).
    depth : int or None
        Pyramid depth.  If None, computed automatically from frame dimensions
        as ``floor(log2(min(H, W))) - 2``.

    Returns
    -------
    np.ndarray
        Shape (T, H, W, 3), dtype float32, range [0.0, 1.0].  BGR channel order.

    Raises
    ------
    ValueError
        On invalid argument values.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if freq_low <= 0:
        raise ValueError(f"freq_low must be positive, got {freq_low}")
    if freq_high <= freq_low:
        raise ValueError(f"freq_high ({freq_high}) must be > freq_low ({freq_low})")
    if freq_high >= fps / 2.0:
        raise ValueError(
            f"freq_high ({freq_high} Hz) must be < Nyquist ({fps/2.0} Hz)"
        )
    if colorspace not in _COLORSPACES:
        raise ValueError(f"colorspace must be one of {_COLORSPACES}, got '{colorspace}'")
    if pyramid_type not in _PYRAMID_TYPES:
        raise ValueError(f"pyramid_type must be one of {_PYRAMID_TYPES}, got '{pyramid_type}'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Phase] Device: {device}")

    T, H, W, _ = frames.shape

    # Pyramid depth
    if depth is None:
        depth = int(np.floor(np.log2(min(H, W))) - 2)
    print(f"[Phase] Pyramid depth: {depth}")

    # Convert to processing colorspace
    proc_frames = _to_processing_space(frames, colorspace)

    # Build filters
    csp = _make_pyramid(pyramid_type, depth)
    filters, _ = csp.get_filters(H, W, cropped=False)
    filters_tensor = torch.tensor(np.array(filters, dtype=np.float32)).to(device)

    num_filters = filters_tensor.shape[0]
    if (num_filters - 2) % batch_size != 0:
        print(
            f"[Phase] WARNING: batch_size={batch_size} does not evenly divide "
            f"the number of band-pass filters ({num_filters - 2}). "
            f"The last batch will be smaller."
        )

    # Bandpass transfer function
    tf = bandpass_filter(freq_low, freq_hi=freq_high, fs=fps, num_taps=T, device=device)

    # Create phase processor
    pb = _PhaseBased(
        sigma=sigma,
        transfer_function=tf,
        phase_mag=alpha,
        attenuate=attenuate,
        ref_idx=ref_idx,
        batch_size=batch_size,
        device=device,
    )

    # Process channels
    if colorspace in ("yiq", "rgb"):
        # All 3 channels independently
        proc_tensor = torch.tensor(proc_frames, dtype=torch.float32).to(device)
        result_tensor = torch.zeros_like(proc_tensor)
        for c in range(3):
            video_dft = get_fft2_batch(proc_tensor[:, :, :, c]).to(device)
            torch.cuda.empty_cache()
            result_tensor[:, :, :, c] = pb.process_single_channel(
                proc_tensor[:, :, :, c], filters_tensor, video_dft
            )
            torch.cuda.empty_cache()
        result_np = result_tensor.cpu().numpy()

    elif colorspace == "luma3":
        # Process Y channel only; keep IQ
        proc_tensor = torch.tensor(proc_frames, dtype=torch.float32).to(device)
        video_dft = get_fft2_batch(proc_tensor[:, :, :, 0]).to(device)
        processed_luma = pb.process_single_channel(
            proc_tensor[:, :, :, 0], filters_tensor, video_dft
        )
        result_np = processed_luma.cpu().numpy()  # (T, H, W)

    else:
        # Single-channel (luma1, gray)
        proc_tensor = torch.tensor(proc_frames, dtype=torch.float32).to(device)
        video_dft = get_fft2_batch(proc_tensor).to(device)
        result_np = pb.process_single_channel(
            proc_tensor, filters_tensor, video_dft
        ).cpu().numpy()

    # Convert back to BGR
    return _to_bgr(result_np, proc_frames, colorspace)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-Based Motion Magnification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to input video.")
    p.add_argument("--output", required=True, help="Path to save amplified video.")
    p.add_argument("--freq-low", type=float, required=True,
                   help="Lower temporal bandpass cutoff in Hz.")
    p.add_argument("--freq-high", type=float, required=True,
                   help="Upper temporal bandpass cutoff in Hz.")
    p.add_argument("--amplify", type=float, required=True,
                   help="Phase amplification factor alpha.")
    p.add_argument("--colorspace", default="luma3", choices=list(_COLORSPACES),
                   help="Processing colorspace.")
    p.add_argument("--pyramid", default="half_octave", choices=list(_PYRAMID_TYPES),
                   help="Steerable pyramid type.")
    p.add_argument("--sigma", type=float, default=2.0,
                   help="Gaussian std-dev for amplitude-weighted blurring (0 = off).")
    p.add_argument("--attenuate", action="store_true",
                   help="Attenuate out-of-band frequencies.")
    p.add_argument("--ref-idx", type=int, default=0,
                   help="Reference frame index for phase-delta computation.")
    p.add_argument("--batch", type=int, default=4,
                   help="Filter batch size for GPU processing.")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Spatial scale factor (< 1 downsamples before processing).")
    p.add_argument("--fps", type=float, default=0.0,
                   help="Override detected FPS (0 = auto-detect).")
    p.add_argument("--depth", type=int, default=0,
                   help="Pyramid depth (0 = auto-compute from frame size).")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    frames, detected_fps = load_video(args.input, scale=args.scale)
    fps = args.fps if args.fps > 0 else detected_fps

    print(
        f"Loaded {frames.shape[0]} frames at {fps:.2f} fps  "
        f"({frames.shape[2]}×{frames.shape[1]} px)"
    )

    result = run_phase_based(
        frames=frames,
        fps=fps,
        freq_low=args.freq_low,
        freq_high=args.freq_high,
        alpha=args.amplify,
        colorspace=args.colorspace,
        pyramid_type=args.pyramid,
        sigma=args.sigma,
        attenuate=args.attenuate,
        ref_idx=args.ref_idx,
        batch_size=args.batch,
        depth=args.depth if args.depth > 0 else None,
    )

    save_video(result, args.output, fps=fps)
    print(f"Saved amplified video → {args.output}")


if __name__ == "__main__":
    main()
