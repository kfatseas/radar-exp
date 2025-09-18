"""Window functions used prior to FFT.

The choice of window influences leakage in the spectral domain.  This
module exposes a simple factory that returns numpy arrays for the
commonly used window types.  Additional parameters (e.g., Chebyshev
attenuation) can be passed via keyword arguments.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from scipy import signal


def get_window(name: str, length: int, **kwargs: float) -> np.ndarray:
    """Return a window of a given type and length.

    Parameters
    ----------
    name : str
        The window type.  Supported values are 'hann', 'hamming',
        'blackman', 'chebwin' and 'rectangular'.
    length : int
        The number of samples in the window.
    **kwargs : float
        Additional parameters forwarded to the underlying SciPy
        implementation.  For example the Chebyshev window accepts an
        `at` parameter defining the sidelobe attenuation in dB.

    Returns
    -------
    np.ndarray
        A one‑dimensional window of the requested type.

    Notes
    -----
    The returned window is normalised to have a mean of one.  This
    simplifies subsequent magnitude normalisation.
    """
    name = name.lower()
    if length <= 0:
        raise ValueError("length must be positive")
    if name in {"hann", "hanning"}:
        win = signal.windows.hann(length, **kwargs)
    elif name == "hamming":
        win = signal.windows.hamming(length, **kwargs)
    elif name == "blackman":
        win = signal.windows.blackman(length, **kwargs)
    elif name == "chebwin":
        # Chebyshev window requires attenuation parameter 'at'.  Use a
        # sensible default if none provided.
        at = float(kwargs.get("at", 80.0))
        win = signal.windows.chebwin(length, at)
    elif name == "rectangular" or name == "rect" or name == "none":
        win = np.ones(length)
    else:
        raise ValueError(f"Unknown window type: {name}")
    # Normalise to unit average to preserve energy across FFT sizes.
    mean_val = np.mean(win)
    if mean_val != 0:
        win = win / mean_val
    return win


def get_2d_window(range_len: int, doppler_len: int, range_window: str = "hann", doppler_window: str = "hann", **params: float) -> Dict[str, np.ndarray]:
    """Construct 1D windows for range and Doppler dimensions.

    Returns a dictionary with keys 'range' and 'doppler' mapping to the
    corresponding window arrays.  Additional parameters (e.g. attenuation
    for Chebyshev windows) can be supplied via `params` and will be
    forwarded to both windows.
    """
    return {
        "range": get_window(range_window, range_len, **params),
        "doppler": get_window(doppler_window, doppler_len, **params),
    }