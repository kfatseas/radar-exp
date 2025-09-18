"""FFT utilities for range and velocity processing.

This module provides functions to compute the range FFT (along the
fast‑time axis) and the velocity FFT (along the slow‑time/chirp axis) of
a radar cube.  Each function accepts an optional window specification
and zero‑padding length.  Magnitude normalisation is performed to
preserve the physical meaning of the returned spectra.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import signal

from .windows import get_window


def range_fft(cube: np.ndarray, window_type: str = "hann", n_fft: Optional[int] = None) -> np.ndarray:
    """Compute the range FFT of a radar cube.

    Parameters
    ----------
    cube : np.ndarray
        The input radar cube with shape `(antennas, chirps, samples)`.
    window_type : str, optional
        The type of window to apply along the sample (fast‑time) axis.
        Defaults to `'hann'`.  See `radar_exp.dsp.windows.get_window` for
        supported values.
    n_fft : int, optional
        The number of FFT points.  If `None` the FFT length equals the
        number of samples.  Zero padding increases frequency resolution.

    Returns
    -------
    np.ndarray
        The range‑processed cube of shape `(antennas, chirps, n_fft//2)`
        containing the positive frequency bins.
    """
    antennas, chirps, samples = cube.shape
    fft_size = n_fft if n_fft is not None else samples
    # Generate and apply window on the sample axis
    win = get_window(window_type, samples)[np.newaxis, np.newaxis, :]
    cube_windowed = cube * win
    # Perform FFT along the sample axis
    spectrum = np.fft.rfft(cube_windowed, n=fft_size, axis=2)
    # Normalise by window and length to preserve magnitude
    spectrum /= (fft_size * (1.0 / np.mean(win)))
    return spectrum


def velocity_fft(range_cube: np.ndarray, window_type: str = "hann", n_fft: Optional[int] = None) -> np.ndarray:
    """Compute the velocity (Doppler) FFT of a range‑processed cube.

    Parameters
    ----------
    range_cube : np.ndarray
        The output of `range_fft`, of shape `(antennas, chirps, samples)`.
    window_type : str, optional
        Window to apply along the chirp (slow‑time) axis.  Defaults to
        `'hann'`.
    n_fft : int, optional
        Number of points in the Doppler FFT.  If `None` uses the number
        of chirps.

    Returns
    -------
    np.ndarray
        The complex range–Doppler cube of shape `(antennas, n_fft, samples)`.
        The Doppler axis is centered (zero frequency at the centre).
    """
    antennas, chirps, samples = range_cube.shape
    fft_size = n_fft if n_fft is not None else chirps
    # Windowing on the slow‑time axis
    win = get_window(window_type, chirps)[np.newaxis, :, np.newaxis]
    cube_windowed = range_cube * win
    # Compute FFT along the chirp axis
    spectrum = np.fft.fft(cube_windowed, n=fft_size, axis=1)
    # Shift zero Doppler to the centre
    spectrum = np.fft.fftshift(spectrum, axes=1)
    # Normalise
    spectrum /= (fft_size * (1.0 / np.mean(win)))
    return spectrum