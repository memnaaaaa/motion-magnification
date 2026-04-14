"""
src/pyramids/spatial.py
-----------------------
Laplacian / Gaussian spatial pyramid construction and collapse.

Used by the Eulerian pipeline to decompose frames into spatial-frequency bands,
filter each band in time, and reconstruct the amplified frame.

Functions
---------
build_laplacian_pyramid(frame, levels)  ->  list[np.ndarray]
collapse_laplacian_pyramid(pyramid)     ->  np.ndarray
"""

from __future__ import annotations

import cv2
import numpy as np


def build_laplacian_pyramid(
    frame: np.ndarray,
    levels: int,
) -> list[np.ndarray]:
    """
    Build a Laplacian pyramid from a single frame.

    Parameters
    ----------
    frame : np.ndarray
        Input frame, dtype float32.
        Shape must be (H, W) or (H, W, C).
    levels : int
        Number of Laplacian difference bands.
        The returned list has length ``levels + 1``.

    Returns
    -------
    list[np.ndarray]
        ``pyramid[0 .. levels-1]`` — Laplacian difference bands
            (band-pass spatial detail), same shape as corresponding
            Gaussian level.
        ``pyramid[levels]`` — coarsest Gaussian residual (low-pass),
            shape roughly (H // 2**levels, W // 2**levels [, C]).

    Raises
    ------
    ValueError
        If ``frame`` is not float32, has wrong ndim, if ``levels < 1``,
        or if the frame is too small to build the requested pyramid depth.

    Notes
    -----
    Each Laplacian level is computed as::

        L[k] = G[k] − pyrUp(G[k+1])   (resized to exact shape of G[k])

    ``cv2.pyrUp`` does not guarantee the output matches the original
    size for odd dimensions, so an explicit ``cv2.resize`` is applied.

    ``cv2.resize`` dsize convention: ``(width, height)`` — NOT
    ``(height, width)``.
    """
    # --- Input validation ------------------------------------------------
    if frame.dtype != np.float32:
        raise ValueError(
            f"frame must be float32, got {frame.dtype}. "
            "Convert with frame.astype(np.float32) before calling."
        )
    if frame.ndim not in (2, 3):
        raise ValueError(
            f"frame must be 2-D (H, W) or 3-D (H, W, C), got ndim={frame.ndim}."
        )
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}.")

    h, w = frame.shape[:2]
    min_dim = min(h, w)
    if min_dim < 2 ** levels:
        raise ValueError(
            f"Frame too small for {levels} pyramid levels: "
            f"minimum dimension is {min_dim}, need at least {2 ** levels}."
        )

    # --- Build Gaussian pyramid -----------------------------------------
    gaussian: list[np.ndarray] = [frame]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))

    # --- Build Laplacian pyramid ----------------------------------------
    pyramid: list[np.ndarray] = []
    for k in range(levels):
        g_k = gaussian[k]
        g_k1 = gaussian[k + 1]

        # pyrUp may produce a size off by ±1 for odd dimensions; fix it.
        target_h, target_w = g_k.shape[:2]
        upsampled = cv2.pyrUp(g_k1)
        if upsampled.shape[:2] != (target_h, target_w):
            upsampled = cv2.resize(
                upsampled,
                (target_w, target_h),   # dsize = (width, height)
                interpolation=cv2.INTER_LINEAR,
            )

        pyramid.append(g_k - upsampled)

    # Coarsest residual (low-pass)
    pyramid.append(gaussian[levels])

    return pyramid


def collapse_laplacian_pyramid(pyramid: list[np.ndarray]) -> np.ndarray:
    """
    Reconstruct a frame from its Laplacian pyramid.

    Parameters
    ----------
    pyramid : list[np.ndarray]
        As returned by ``build_laplacian_pyramid``.  The last element is
        the coarsest Gaussian residual; all preceding elements are
        Laplacian difference bands.

    Returns
    -------
    np.ndarray
        Reconstructed frame, shape == ``pyramid[0].shape``, dtype float32.
        Round-trip atol vs the original input is approximately 1e-5 (due
        to float32 accumulation).

    Raises
    ------
    ValueError
        If ``pyramid`` is empty or contains fewer than 2 levels.

    Notes
    -----
    Reconstruction iterates from the coarsest level upward::

        result = pyramid[-1]
        for k in reversed(range(levels)):
            result = pyrUp(result) [resized to pyramid[k].shape] + pyramid[k]
    """
    if len(pyramid) < 2:
        raise ValueError(
            f"pyramid must have at least 2 entries (1 Laplacian band + residual), "
            f"got {len(pyramid)}."
        )

    result: np.ndarray = pyramid[-1].astype(np.float32)

    for k in range(len(pyramid) - 2, -1, -1):
        target_h, target_w = pyramid[k].shape[:2]
        upsampled = cv2.pyrUp(result)
        if upsampled.shape[:2] != (target_h, target_w):
            upsampled = cv2.resize(
                upsampled,
                (target_w, target_h),   # dsize = (width, height)
                interpolation=cv2.INTER_LINEAR,
            )
        result = upsampled + pyramid[k]

    return result
