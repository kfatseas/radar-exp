"""Range–Doppler map computation.

The main entry point in this module is `compute_range_doppler_map`,
which applies range and velocity FFTs to a raw radar cube and
aggregates over antennas to form a 2‑D range–Doppler map.  The
function accepts a configuration dictionary specifying window types,
FFT lengths, aggregation methods and whether to return linear or
logarithmic magnitudes.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .fft import range_fft, velocity_fft


def magnitude_db(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute magnitude of a complex array in decibels.

    Adds a small epsilon before taking the log to avoid log of zero.
    """
    return 20.0 * np.log10(np.abs(x) + eps)


def magnitude_linear(x: np.ndarray) -> np.ndarray:
    """Compute linear magnitude of a complex array."""
    return np.abs(x)


def compute_range_doppler_map(
    cube: np.ndarray,
    windows: Dict[str, str] | None = None,
    fft_sizes: Dict[str, int | None] | None = None,
    aggregate: str = "max",
    magnitude: str = "db",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the range–Doppler map and return both the map and the complex cube.

    Parameters
    ----------
    cube : np.ndarray
        Raw radar cube of shape `(antennas, chirps, samples)`.
    windows : dict, optional
        Dictionary with keys `range` and `doppler` specifying window
        types to apply along the corresponding axes.  Defaults to
        `{'range': 'hann', 'doppler': 'hann'}`.
    fft_sizes : dict, optional
        Dictionary with keys `range` and `doppler` defining the FFT
        lengths.  Use `None` to select the natural length.  Defaults
        to `None` for both axes.
    aggregate : str, optional
        How to combine spectra from multiple antennas.  Supported values
        are `'max'` (default) and `'sum'`.
    magnitude : str, optional
        Return magnitude either in `'db'` (decibels) or `'linear'`.

    Returns
    -------
    tuple
        `(rd_map, rd_cube)` where `rd_map` is a 2‑D array with axes
        `[range_bin, doppler_bin]` and `rd_cube` is the complex cube
        after FFT processing with shape `(antennas, doppler_bins, range_bins)`.
    """
    # Set default window and FFT length
    if windows is None:
        windows = {"range": "hann", "doppler": "hann"}
    if fft_sizes is None:
        fft_sizes = {"range": None, "doppler": None}
    # Perform range FFT
    range_cube = range_fft(
        cube,
        window_type=windows.get("range", "hann"),
        n_fft=fft_sizes.get("range"),
    )
    # Perform velocity FFT
    rd_cube = velocity_fft(
        range_cube,
        window_type=windows.get("doppler", "hann"),
        n_fft=fft_sizes.get("doppler"),
    )
    # Aggregate over antennas
    if aggregate == "max":
        rd_agg = np.max(rd_cube, axis=0)
    elif aggregate == "sum":
        rd_agg = np.sum(rd_cube, axis=0)
    else:
        raise ValueError(f"Unsupported aggregation: {aggregate}")
    # Compute magnitude.  Transpose so that the first axis is range and the
    # second axis is Doppler, matching the conventions used by detection
    # routines.  Internally `rd_agg` has shape (doppler, range).
    if magnitude == "db":
        rd_map = magnitude_db(rd_agg).T
    elif magnitude == "linear":
        rd_map = magnitude_linear(rd_agg).T
    else:
        raise ValueError(f"Unsupported magnitude mode: {magnitude}")
    return rd_map, rd_cube