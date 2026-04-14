"""
src/filters/temporal.py — Temporal bandpass filtering for 1-D time-series.

Two public functions:

  bandpass_fir   — zero-phase FIR filter via FFT multiplication.
                   Use this for the phase-based pipeline: FIR preserves phase,
                   which is critical when the signal being filtered IS phase.

  bandpass_iir   — zero-phase IIR (Butterworth) filter via scipy.signal.filtfilt.
                   Simpler and faster for the Eulerian pipeline where the output
                   is intensity, not phase.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, firwin


def bandpass_filter_1d(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
    fps: float,
) -> np.ndarray:
    """Zero-phase FIR bandpass filter applied via FFT multiplication.

    Use this for the phase-based pipeline.  IIR filters introduce phase
    distortion that corrupts the phase signal being manipulated; FIR avoids that.

    Args:
        signal:   shape (T,), float32 or float64 — the 1-D time series to filter.
        low_hz:   Lower cutoff frequency in Hz (exclusive).
        high_hz:  Upper cutoff frequency in Hz (exclusive).
        fps:      Video frame rate in Hz (samples per second).

    Returns:
        Filtered signal, shape (T,), same dtype as input.

    Raises:
        ValueError: If cutoff frequencies are out of range or ordering is wrong.
    """
    if low_hz <= 0.0:
        raise ValueError(f"low_hz must be positive, got {low_hz}")
    if high_hz <= low_hz:
        raise ValueError(f"high_hz ({high_hz}) must be greater than low_hz ({low_hz})")
    nyq = fps / 2.0
    if high_hz >= nyq:
        raise ValueError(
            f"high_hz ({high_hz} Hz) must be less than Nyquist ({nyq} Hz) for fps={fps}"
        )

    n = len(signal)
    taps = firwin(
        numtaps=n,
        cutoff=[low_hz / nyq, high_hz / nyq],
        pass_zero=False,
    )
    H = np.fft.fft(np.fft.ifftshift(taps))
    S = np.fft.fft(signal.astype(np.float64))
    filtered = np.fft.ifft(H * S).real
    return filtered.astype(signal.dtype)


def bandpass_filter_butter(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
    fps: float,
    order: int = 5,
) -> np.ndarray:
    """Zero-phase IIR Butterworth bandpass filter via forward-backward filtering.

    Use this for the Eulerian pipeline where phase distortion is not a concern
    and the simpler IIR design is sufficient.

    Args:
        signal:   shape (T,), float32 or float64.
        low_hz:   Lower cutoff frequency in Hz.
        high_hz:  Upper cutoff frequency in Hz.
        fps:      Video frame rate in Hz.
        order:    Butterworth filter order (default 5).

    Returns:
        Filtered signal, shape (T,), same dtype as input.

    Raises:
        ValueError: If cutoff frequencies are out of range or ordering is wrong.
    """
    if low_hz <= 0.0:
        raise ValueError(f"low_hz must be positive, got {low_hz}")
    if high_hz <= low_hz:
        raise ValueError(f"high_hz ({high_hz}) must be greater than low_hz ({low_hz})")
    nyq = fps / 2.0
    if high_hz >= nyq:
        raise ValueError(
            f"high_hz ({high_hz} Hz) must be less than Nyquist ({nyq} Hz) for fps={fps}"
        )

    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    filtered = filtfilt(b, a, signal.astype(np.float64))
    return filtered.astype(signal.dtype)
