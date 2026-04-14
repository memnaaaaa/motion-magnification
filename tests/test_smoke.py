"""
tests/test_smoke.py
-------------------
Smoke tests for all src/ modules.

These tests use only synthetic data (no external files) so they run
in any environment after ``pip install -r requirements.txt``.

Run with:
    conda activate motionmag
    python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_video(T: int = 30, H: int = 64, W: int = 64, C: int = 3) -> np.ndarray:
    """Create a synthetic float32 BGR video in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.random((T, H, W, C), dtype=np.float64).astype(np.float32)


def _make_sinusoidal_video(
    T: int = 60,
    H: int = 64,
    W: int = 64,
    signal_hz: float = 1.0,
    fps: float = 30.0,
) -> np.ndarray:
    """Create a video with a sinusoidal intensity pulse at ``signal_hz``."""
    t = np.arange(T) / fps
    pulse = 0.05 * np.sin(2 * np.pi * signal_hz * t)  # small oscillation

    frames = np.full((T, H, W, 3), 0.5, dtype=np.float32)
    frames[:, H // 4: H // 2, W // 4: W // 2, :] += pulse[:, None, None, None]
    return np.clip(frames, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_stabilize(self):
        from src.stabilization.raft_stabilize import stabilize_video
        assert callable(stabilize_video)

    def test_import_video_io(self):
        from src.io.video_io import load_video, save_video
        assert callable(load_video)
        assert callable(save_video)

    def test_import_temporal(self):
        from src.filters.temporal import bandpass_filter_1d, bandpass_filter_butter
        assert callable(bandpass_filter_1d)
        assert callable(bandpass_filter_butter)

    def test_import_spatial(self):
        from src.pyramids.spatial import build_laplacian_pyramid, collapse_laplacian_pyramid
        assert callable(build_laplacian_pyramid)
        assert callable(collapse_laplacian_pyramid)

    def test_import_pyramid_utils(self):
        from src.pyramids.pyramid_utils import (
            get_polar_grid, get_filter_crops, get_cropped_filters,
            build_level, recon_level, build_level_batch, recon_level_batch,
        )
        for fn in (get_polar_grid, get_filter_crops, get_cropped_filters,
                   build_level, recon_level, build_level_batch, recon_level_batch):
            assert callable(fn)

    def test_import_phase_utils(self):
        from src.utils.phase_utils import (
            rgb2yiq, bgr2yiq, yiq2rgb, get_fft2_batch, bandpass_filter
        )
        for fn in (rgb2yiq, bgr2yiq, yiq2rgb, get_fft2_batch, bandpass_filter):
            assert callable(fn)

    def test_import_steerable(self):
        from src.pyramids.steerable import SteerablePyramid, SuboctaveSP
        assert SteerablePyramid
        assert SuboctaveSP

    def test_import_eulerian(self):
        from src.magnification.eulerian import run_eulerian
        assert callable(run_eulerian)

    def test_import_phase_based(self):
        from src.magnification.phase_based import run_phase_based
        assert callable(run_phase_based)

    def test_import_render(self):
        from src.visualization.render import render_side_by_side, write_heatmap_overlay
        assert callable(render_side_by_side)
        assert callable(write_heatmap_overlay)

    def test_import_metrics(self):
        from src.utils.metrics import compute_temporal_snr, snr_alpha_sweep
        assert callable(compute_temporal_snr)
        assert callable(snr_alpha_sweep)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

class TestVideoIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        from src.io.video_io import load_video, save_video

        frames = _make_video(T=10, H=32, W=32)
        out_path = tmp_path / "test_video.mp4"
        save_video(frames, str(out_path), fps=10.0)

        loaded, fps = load_video(str(out_path))
        assert loaded.dtype == np.float32
        assert loaded.ndim == 4
        assert loaded.shape[0] == 10
        assert fps > 0.0

    def test_save_video_invalid_shape(self, tmp_path):
        from src.io.video_io import save_video
        with pytest.raises(ValueError):
            save_video(np.zeros((10, 32, 32), dtype=np.float32), str(tmp_path / "x.mp4"), fps=10.0)

    def test_load_video_missing_file(self):
        from src.io.video_io import load_video
        with pytest.raises(FileNotFoundError):
            load_video("does_not_exist.mp4")


# ---------------------------------------------------------------------------
# Temporal filter
# ---------------------------------------------------------------------------

class TestTemporalFilter:
    def test_fir_output_shape_and_dtype(self):
        from src.filters.temporal import bandpass_filter_1d
        sig = np.random.default_rng(0).random(100).astype(np.float32)
        out = bandpass_filter_1d(sig, low_hz=0.5, high_hz=4.0, fps=30.0)
        assert out.shape == sig.shape
        assert out.dtype == np.float32

    def test_iir_output_shape_and_dtype(self):
        from src.filters.temporal import bandpass_filter_butter
        sig = np.random.default_rng(1).random(100).astype(np.float32)
        out = bandpass_filter_butter(sig, low_hz=0.5, high_hz=4.0, fps=30.0)
        assert out.shape == sig.shape
        assert out.dtype == np.float32

    def test_fir_invalid_freq(self):
        from src.filters.temporal import bandpass_filter_1d
        sig = np.ones(50, dtype=np.float32)
        with pytest.raises(ValueError):
            bandpass_filter_1d(sig, low_hz=20.0, high_hz=4.0, fps=30.0)  # high < low


# ---------------------------------------------------------------------------
# Laplacian pyramid
# ---------------------------------------------------------------------------

class TestLaplacianPyramid:
    def test_build_and_collapse_roundtrip(self):
        from src.pyramids.spatial import build_laplacian_pyramid, collapse_laplacian_pyramid
        frame = np.random.default_rng(42).random((64, 64)).astype(np.float32)
        pyr = build_laplacian_pyramid(frame, levels=3)
        assert len(pyr) == 4  # 3 Laplacian bands + 1 residual
        reconstructed = collapse_laplacian_pyramid(pyr)
        assert reconstructed.shape == frame.shape
        np.testing.assert_allclose(reconstructed, frame, atol=1e-4)

    def test_build_requires_float32(self):
        from src.pyramids.spatial import build_laplacian_pyramid
        with pytest.raises(ValueError):
            build_laplacian_pyramid(np.zeros((64, 64), dtype=np.float64), levels=2)


# ---------------------------------------------------------------------------
# Phase utilities
# ---------------------------------------------------------------------------

class TestPhaseUtils:
    def test_bgr2yiq_yiq2rgb_roundtrip(self):
        from src.utils.phase_utils import bgr2yiq, yiq2rgb
        import cv2
        bgr = np.random.default_rng(7).random((32, 32, 3)).astype(np.float32)
        yiq = bgr2yiq(bgr)
        rgb = yiq2rgb(yiq)
        # Convert back to BGR for comparison
        bgr_reconstructed = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        np.testing.assert_allclose(bgr, bgr_reconstructed, atol=0.02)

    def test_bgr2yiq_output_dtype(self):
        from src.utils.phase_utils import bgr2yiq
        bgr = np.ones((8, 8, 3), dtype=np.float32) * 0.5
        yiq = bgr2yiq(bgr)
        assert yiq.dtype == np.float32
        assert yiq.shape == (8, 8, 3)

    def test_yiq2rgb_clipped(self):
        from src.utils.phase_utils import yiq2rgb
        # extreme YIQ values should clip to [0, 1]
        yiq = np.array([[[10.0, 10.0, 10.0]]], dtype=np.float32)
        rgb = yiq2rgb(yiq)
        assert rgb.max() <= 1.0 + 1e-6
        assert rgb.min() >= -1e-6


# ---------------------------------------------------------------------------
# Steerable pyramid
# ---------------------------------------------------------------------------

class TestSteerablePyramid:
    def test_get_filters_output_count(self):
        from src.pyramids.steerable import SteerablePyramid
        csp = SteerablePyramid(depth=2, orientations=4, filters_per_octave=1,
                               twidth=1.0, complex_pyr=True)
        filters, crops = csp.get_filters(h=64, w=64)
        # 1 hi-pass + depth * orientations band filters + 1 lo-pass
        expected = 1 + 2 * 4 + 1
        assert len(filters) == expected
        assert len(crops) == expected

    def test_suboctave_get_filters(self):
        from src.pyramids.steerable import SuboctaveSP
        csp = SuboctaveSP(depth=2, orientations=4, filters_per_octave=2,
                          cos_order=6, complex_pyr=True)
        filters, crops = csp.get_filters(h=64, w=64)
        assert len(filters) > 0


# ---------------------------------------------------------------------------
# Eulerian pipeline (small synthetic video)
# ---------------------------------------------------------------------------

class TestEulerianPipeline:
    def test_output_shape_and_range(self):
        from src.magnification.eulerian import run_eulerian
        frames = _make_sinusoidal_video(T=60, H=32, W=32, signal_hz=1.0, fps=30.0)
        result = run_eulerian(
            frames=frames,
            fps=30.0,
            freq_low=0.5,
            freq_high=4.0,
            alpha=10.0,
            levels=2,
        )
        assert result.shape == frames.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0 - 1e-5
        assert result.max() <= 1.0 + 1e-5

    def test_invalid_alpha(self):
        from src.magnification.eulerian import run_eulerian
        with pytest.raises(ValueError):
            run_eulerian(
                frames=_make_video(T=30, H=32, W=32),
                fps=30.0, freq_low=0.5, freq_high=4.0,
                alpha=-1.0, levels=2,
            )


# ---------------------------------------------------------------------------
# Side-by-side render
# ---------------------------------------------------------------------------

class TestRender:
    def test_side_by_side_shape(self):
        from src.visualization.render import render_side_by_side
        orig = _make_video(T=5, H=32, W=32)
        amp = _make_video(T=5, H=32, W=32)
        combined = render_side_by_side(orig, amp, sep_width=4)
        assert combined.shape == (5, 32, 32 * 2 + 4, 3)

    def test_frame_count_mismatch(self):
        from src.visualization.render import render_side_by_side
        with pytest.raises(ValueError):
            render_side_by_side(_make_video(T=5), _make_video(T=10))

    def test_resize_amplified_to_original(self):
        from src.visualization.render import render_side_by_side
        orig = _make_video(T=5, H=64, W=64)
        amp = _make_video(T=5, H=32, W=32)  # different size
        result = render_side_by_side(orig, amp, sep_width=2)
        assert result.shape[1] == 64  # height matches original


# ---------------------------------------------------------------------------
# Temporal SNR
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_snr_returns_float(self):
        from src.utils.metrics import compute_temporal_snr
        frames = _make_sinusoidal_video(T=60, H=32, W=32, signal_hz=1.0, fps=30.0)
        snr = compute_temporal_snr(
            frames=frames,
            fps=30.0,
            freq_low=0.5,
            freq_high=4.0,
            signal_roi=(8, 8, 24, 24),
            noise_roi=(0, 0, 4, 4),
        )
        assert isinstance(snr, float)

    def test_snr_invalid_roi(self):
        from src.utils.metrics import compute_temporal_snr
        frames = _make_video(T=30, H=32, W=32)
        with pytest.raises(ValueError):
            compute_temporal_snr(
                frames=frames,
                fps=30.0,
                freq_low=0.5,
                freq_high=4.0,
                signal_roi=(0, 0, 100, 100),  # out of bounds
                noise_roi=(0, 0, 8, 8),
            )
