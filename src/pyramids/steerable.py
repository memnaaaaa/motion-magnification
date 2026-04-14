"""
src/pyramids/steerable.py
-------------------------
Regular, Complex, and Sub-octave Complex Steerable Pyramids.

Adapted from the root-level ``steerable_pyramid.py`` reference implementation.
Math is unchanged; imports updated to use ``src.pyramids.pyramid_utils``
and the interface cleaned up with type hints and docstrings.

Sources
-------
Papers:
    - http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf
    - https://www.cns.nyu.edu/pub/eero/simoncelli95b.pdf
Code:
    - http://people.csail.mit.edu/nwadhwa/phase-video/
    - https://github.com/LabForComputationalVision/matlabPyrTools

Classes
-------
SteerablePyramid   — standard steerable pyramid (single/full-octave bandwidth)
SuboctaveSP        — smooth sub-octave complex steerable pyramid
"""

from __future__ import annotations

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt

from src.pyramids.pyramid_utils import get_polar_grid, get_filter_crops


# ---------------------------------------------------------------------------
# Standard steerable pyramid
# ---------------------------------------------------------------------------

class SteerablePyramid:
    """Standard complex steerable pyramid.

    Parameters
    ----------
    depth : int
        Pyramid depth (number of radial frequency bands per octave grouping).
    orientations : int
        Number of orientation bands per radial level.
    filters_per_octave : int
        Number of filters within each octave.  1 = full-octave bandwidth;
        2 = half-octave bandwidth (recommended for phase-based magnification).
    twidth : float
        Transition width between the low- and high-pass filter boundaries.
        Smaller values give a sharper rolloff.
    complex_pyr : bool
        If True, build a complex-valued pyramid (required for phase extraction).
    """

    def __init__(
        self,
        depth: int,
        orientations: int,
        filters_per_octave: int = 1,
        twidth: float = 1.0,
        complex_pyr: bool = False,
    ) -> None:
        self.depth = depth
        self.orientations = orientations
        self.twidth = twidth
        self.complex_pyr = complex_pyr

        # Total number of band-pass filters (excludes hi- and lo-pass)
        self.num_filts = depth * filters_per_octave

        # Bandwidth in octaves per filter
        self.octave_bw = 1.0 / filters_per_octave

    # ------------------------------------------------------------------
    # Private filter builders
    # ------------------------------------------------------------------

    def _get_radial_mask(
        self,
        radius: np.ndarray,
        r: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute paired radial low- and high-pass filter masks.

        Parameters
        ----------
        radius : np.ndarray
            Radial frequency grid, shape (H, W).
        r : float
            Filter boundary radius (in log2 units).

        Returns
        -------
        lo_mask : np.ndarray
            Low-pass filter mask, shape (H, W).
        hi_mask : np.ndarray
            High-pass filter mask, shape (H, W).
        """
        log_rad = np.log2(radius) - np.log2(r)

        hi_mask = np.clip(log_rad, -self.twidth, 0)
        hi_mask = np.abs(np.cos(hi_mask * np.pi / (2 * self.twidth)))
        lo_mask = np.sqrt(1.0 - hi_mask ** 2)

        return lo_mask, hi_mask

    def _get_angle_mask(self, angle: np.ndarray, b: int) -> np.ndarray:
        """Compute an oriented angular filter mask for band ``b``.

        Parameters
        ----------
        angle : np.ndarray
            Angular frequency grid, shape (H, W), in radians.
        b : int
            Orientation index (0-based).

        Returns
        -------
        np.ndarray
            Angular filter mask, shape (H, W).
        """
        order = self.orientations - 1
        const = (
            np.power(2, 2 * order)
            * np.power(factorial(order), 2)
            / (self.orientations * factorial(2 * order))
        )
        angle_mod = np.mod(np.pi + angle - np.pi * b / self.orientations, 2 * np.pi) - np.pi

        if self.complex_pyr:
            # Single lobe — exploits conjugate symmetry of complex pyramid
            return 2 * np.sqrt(const) * np.power(np.cos(angle_mod), order) * (np.abs(angle_mod) < np.pi / 2)
        else:
            return np.abs(2 * np.sqrt(const) * np.power(np.cos(angle_mod), order))

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    def get_filters(
        self,
        h: int,
        w: int,
        cropped: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Build pyramid filters for image dimensions ``(h, w)``.

        Parameters
        ----------
        h : int
            Image height in pixels.
        w : int
            Image width in pixels.
        cropped : bool
            If True, return filters cropped to their non-zero support.
            If False (default), return full-size filters (shape ``(h, w)``
            for each).

        Returns
        -------
        filters : list[np.ndarray]
            Pyramid filters ordered as:
            ``[hi_pass, band_0_orient_0, …, band_N_orient_M, lo_pass]``.
        crops : list[np.ndarray]
            Corresponding crop boxes ``[r0, r1, c0, c1]`` for each filter.
        """
        angle, radius = get_polar_grid(h, w)

        radial_vals = 2.0 ** np.arange(-self.depth, self.octave_bw, self.octave_bw)[::-1]

        lo_mask_prev, hi_mask = self._get_radial_mask(radius, r=radial_vals[0])

        crop = get_filter_crops(hi_mask)
        crops = [crop]
        filters: list[np.ndarray] = [
            hi_mask[crop[0]:crop[1], crop[2]:crop[3]] if cropped else hi_mask
        ]

        for idx, rval in enumerate(radial_vals[1:]):
            lo_mask, hi_mask = self._get_radial_mask(radius, rval)
            rad_mask = hi_mask * lo_mask_prev

            if idx > 0:
                crop = get_filter_crops(rad_mask)

            for b in range(self.orientations):
                angle_mask = self._get_angle_mask(angle, b)
                filt = rad_mask * angle_mask / 2

                if cropped:
                    filters.append(filt[crop[0]:crop[1], crop[2]:crop[3]])
                else:
                    filters.append(filt)

                crops.append(crop)

            lo_mask_prev = lo_mask

        crop = get_filter_crops(lo_mask)
        crops.append(crop)
        filters.append(lo_mask[crop[0]:crop[1], crop[2]:crop[3]] if cropped else lo_mask)

        return filters, crops

    # ------------------------------------------------------------------
    # Pyramid build / reconstruct
    # ------------------------------------------------------------------

    def build_pyramid(
        self,
        image: np.ndarray,
        cropped_filters: list[np.ndarray],
        crops: list[np.ndarray],
        freq: bool = False,
    ) -> list[np.ndarray]:
        """Decompose a single-channel image into pyramid subbands.

        Parameters
        ----------
        image : np.ndarray
            Shape (H, W), float32.  Single-channel input image.
        cropped_filters : list[np.ndarray]
            Cropped filter arrays as returned by :meth:`get_filters`.
        crops : list[np.ndarray]
            Crop indices as returned by :meth:`get_filters`.
        freq : bool
            If True, return subbands in the frequency domain (DFT).

        Returns
        -------
        list[np.ndarray]
            One array per filter; each is the complex subband response
            (or its DFT if ``freq=True``).
        """
        image_dft = np.fft.fftshift(np.fft.fft2(image))

        pyramid = []
        for filt, crop in zip(cropped_filters, crops):
            dft = image_dft[crop[0]:crop[1], crop[2]:crop[3]] * filt

            if freq:
                pyramid.append(dft)
            elif self.complex_pyr:
                pyramid.append(np.fft.ifft2(np.fft.ifftshift(dft)))
            else:
                pyramid.append(np.fft.ifft2(np.fft.ifftshift(dft)).real)

        return pyramid

    def build_pyramid_full(
        self,
        image: np.ndarray,
        filters: np.ndarray,
        freq: bool = False,
    ) -> np.ndarray:
        """Vectorized pyramid decomposition with uncropped filters.

        Parameters
        ----------
        image : np.ndarray
            Shape (H, W), float32.
        filters : np.ndarray
            Shape (num_filters, H, W).  Uncropped filters stacked along axis 0.
        freq : bool
            Return frequency-domain subbands if True.

        Returns
        -------
        np.ndarray
            Shape (num_filters, H, W).  Complex subbands (or their DFTs).
        """
        image_dft = np.fft.fftshift(np.fft.fft2(image))[np.newaxis, :, :]
        dft = image_dft * filters

        if freq:
            return dft

        if self.complex_pyr:
            return np.fft.ifft2(np.fft.ifftshift(dft, axes=(1, 2)))
        else:
            return np.fft.ifft2(np.fft.ifftshift(dft, axes=(1, 2))).real

    def reconstruct_image_dft(
        self,
        pyramid: list[np.ndarray],
        cropped_filters: list[np.ndarray],
        crops: list[np.ndarray],
        freq: bool = False,
    ) -> np.ndarray:
        """Reconstruct the full-resolution image DFT from pyramid subbands.

        Parameters
        ----------
        pyramid : list[np.ndarray]
            Subbands as returned by :meth:`build_pyramid`.
        cropped_filters : list[np.ndarray]
            Same filters used during decomposition.
        crops : list[np.ndarray]
            Same crop indices used during decomposition.
        freq : bool
            If True, the pyramid subbands are already in the frequency domain.

        Returns
        -------
        np.ndarray
            Shape (H, W), complex128.  Reconstructed DFT.
        """
        h, w = pyramid[0].shape
        recon_dft = np.zeros((h, w), dtype=np.complex128)

        for i, (pyr, filt, crop) in enumerate(zip(pyramid, cropped_filters, crops)):
            dft = pyr if freq else np.fft.fftshift(np.fft.fft2(pyr))

            if self.complex_pyr and (0 < i < len(cropped_filters) - 1):
                recon_dft[crop[0]:crop[1], crop[2]:crop[3]] += 2.0 * dft * filt
            else:
                recon_dft[crop[0]:crop[1], crop[2]:crop[3]] += dft * filt

        return recon_dft

    def reconstruct_image_dft_full(
        self,
        pyramid: list[np.ndarray],
        filters: np.ndarray,
        freq: bool = False,
    ) -> np.ndarray:
        """Reconstruct image DFT from pyramid subbands using full (uncropped) filters.

        Parameters
        ----------
        pyramid : list[np.ndarray]
            Subbands as returned by :meth:`build_pyramid_full`.
        filters : np.ndarray
            Uncropped filters, shape (num_filters, H, W).
        freq : bool
            If True, the pyramid subbands are already in the frequency domain.

        Returns
        -------
        np.ndarray
            Shape (H, W), complex128.
        """
        h, w = pyramid[0].shape
        recon_dft = np.zeros((h, w), dtype=np.complex128)

        for i, (pyr, filt) in enumerate(zip(pyramid, filters)):
            dft = pyr if freq else np.fft.fftshift(np.fft.fft2(pyr))

            if self.complex_pyr and (0 < i < len(filters) - 1):
                recon_dft += 2.0 * dft * filt
            else:
                recon_dft += dft * filt

        return recon_dft

    def reconstruct_image(
        self,
        pyramid: list[np.ndarray],
        filters: list[np.ndarray] | np.ndarray,
        crops: list[np.ndarray] | None = None,
        full: bool = False,
        freq: bool = False,
    ) -> np.ndarray:
        """Reconstruct a spatial-domain image from pyramid subbands.

        Parameters
        ----------
        pyramid : list[np.ndarray]
            Subbands.
        filters : list[np.ndarray] or np.ndarray
            Cropped (list) or full (ndarray, requires ``full=True``) filters.
        crops : list[np.ndarray] or None
            Crop indices; required when ``full=False``.
        full : bool
            Use full (uncropped) filters if True.
        freq : bool
            Subbands are in the frequency domain if True.

        Returns
        -------
        np.ndarray
            Shape (H, W), float64.  Reconstructed image.
        """
        if full:
            recon_dft = self.reconstruct_image_dft_full(pyramid, filters, freq)
        else:
            recon_dft = self.reconstruct_image_dft(pyramid, filters, crops, freq)

        return np.fft.ifft2(np.fft.ifftshift(recon_dft)).real

    def display(
        self,
        filters: list[np.ndarray],
        title: str = "",
    ) -> tuple:
        """Display all band-pass pyramid filters (excludes hi- and lo-pass).

        Parameters
        ----------
        filters : list[np.ndarray]
            Filter list as returned by :meth:`get_filters`.
        title : str
            Figure title.

        Returns
        -------
        tuple
            ``(fig, ax)`` matplotlib handles.
        """
        fig, ax = plt.subplots(self.num_filts, self.orientations, figsize=(30, 20))
        fig.suptitle(title, size=22)

        for i in range(self.num_filts):
            idx = i * self.orientations
            for j in range(1, self.orientations + 1):
                ax[i][j - 1].imshow(filters[idx + j])

        plt.tight_layout()
        return fig, ax


# ---------------------------------------------------------------------------
# Sub-octave smooth-window pyramid
# ---------------------------------------------------------------------------

class SuboctaveSP(SteerablePyramid):
    """Smooth sub-octave complex steerable pyramid.

    Uses a cosine-windowed radial filter bank for smoother frequency tiling
    and better reconstruction.

    Parameters
    ----------
    depth : int
        Pyramid depth.
    orientations : int
        Number of orientations per level.
    filters_per_octave : int
        Number of radial filters per octave (2 = half-octave, 4 = quarter-octave).
    cos_order : int
        Order of the cosine smoothing function (default 6).
    complex_pyr : bool
        Build complex-valued pyramid (required for phase extraction).
    """

    def __init__(
        self,
        depth: int,
        orientations: int,
        filters_per_octave: int,
        cos_order: int = 6,
        complex_pyr: bool = True,
    ) -> None:
        self.depth = depth
        self.num_filts = depth * filters_per_octave
        self.orientations = orientations
        self.filters_per_octave = filters_per_octave
        self.cos_order = cos_order
        self.complex_pyr = complex_pyr

    def _get_angle_mask_smooth(self, angle: np.ndarray, b: int) -> np.ndarray:
        """Compute a smooth oriented angular mask for band ``b``.

        Parameters
        ----------
        angle : np.ndarray
            Angular frequency grid, shape (H, W).
        b : int
            Orientation index.

        Returns
        -------
        np.ndarray
            Angular filter mask, shape (H, W).
        """
        order = self.orientations - 1
        const = (
            np.power(2, 2 * order)
            * np.power(factorial(order), 2)
            / (self.orientations * factorial(2 * order))
        )
        angle_mod = np.mod(np.pi + angle - np.pi * b / self.orientations, 2 * np.pi) - np.pi

        return (
            np.sqrt(const)
            * np.power(np.cos(angle_mod), order)
            * (np.abs(angle_mod) < np.pi / 2)
        )

    @staticmethod
    def window_func(x: np.ndarray, center: float) -> np.ndarray:
        """Rectangular window centred at ``center`` with half-width π/2."""
        return np.abs(x - center) < np.pi / 2

    def get_filters(
        self,
        h: int,
        w: int,
        cropped: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Build sub-octave pyramid filters for image dimensions ``(h, w)``.

        Parameters
        ----------
        h : int
            Image height.
        w : int
            Image width.
        cropped : bool
            Return filters cropped to their non-zero support if True.

        Returns
        -------
        filters : list[np.ndarray]
            Ordered ``[hi_pass, subbands…, lo_pass]``.
        crops : list[np.ndarray]
            Crop indices for each filter.
        """
        angle, radius = get_polar_grid(h, w)

        # Build log-scaled radial coordinate
        rad = np.log2(radius)
        rad = (self.depth + rad) / self.depth
        rad = rad * (np.pi / 2 + np.pi / 7 * self.num_filts)

        # Radial cosine-windowed filters
        const = (
            np.power(2, 2 * self.cos_order)
            * np.power(factorial(self.cos_order), 2)
            / ((self.cos_order + 1) * factorial(2 * self.cos_order))
        )
        rad_filters: list[np.ndarray] = []
        total = np.zeros((h, w))

        for k in reversed(range(self.num_filts)):
            shift = np.pi / (self.cos_order + 1) * (k + 1) + 2 * np.pi / 7
            rf = (
                np.sqrt(const)
                * np.power(np.cos(rad - shift), self.cos_order)
                * self.window_func(rad, shift)
            )
            rad_filters.append(rf)
            total += rf ** 2

        # Low- and high-pass filters
        dims = np.array([h, w])
        center = np.ceil(dims / 2).astype(int)
        lodims = np.ceil(center / 4).astype(int)

        r0, r1 = center[0] - lodims[0], center[0] + lodims[0]
        c0, c1 = center[1] - lodims[1], center[1] + lodims[1]

        lopass = np.zeros((h, w))
        lopass[r0:r1, c0:c1] = np.sqrt(np.abs(1 - total[r0:r1, c0:c1]))
        hipass = np.sqrt(np.abs(1 - (total + lopass ** 2)))

        # Oriented angle masks
        angle_masks = [self._get_angle_mask_smooth(angle, b) for b in range(self.orientations)]

        # Assemble filter list
        filters: list[np.ndarray] = [hipass]
        crops: list[np.ndarray] = [get_filter_crops(hipass)]

        for rf in rad_filters:
            for ang in angle_masks:
                filt = rf * ang
                crop = get_filter_crops(filt)
                crops.append(crop)
                filters.append(filt[crop[0]:crop[1], crop[2]:crop[3]] if cropped else filt)

        crop = get_filter_crops(lopass)
        crops.append(crop)
        filters.append(lopass[crop[0]:crop[1], crop[2]:crop[3]] if cropped else lopass)

        return filters, crops
