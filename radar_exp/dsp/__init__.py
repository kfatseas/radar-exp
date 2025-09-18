"""Signal processing primitives for radar cube processing.

Modules in this package implement window functions, FFT routines,
range–Doppler map computation, constant false alarm rate (CFAR)
detection and post‑processing utilities.  All functions include type
hints and simple docstrings to aid readability.
"""

from . import windows, fft, rd_map, cfar, postproc

__all__ = ["windows", "fft", "rd_map", "cfar", "postproc"]