"""Constant false alarm rate (CFAR) detection algorithms.

This module implements simple CFAR mechanisms to produce a binary
detection mask from a range–Doppler map.  Two approaches are
supported:

* **Percentile threshold**: selects all cells above a specified
  percentile of the global RD map distribution.  This corresponds to
  the behaviour of the original script which used the 99th percentile.

* **CA CFAR (cell‑averaging)**: computes the local noise estimate by
  averaging reference cells in a square neighbourhood while excluding
  guard cells around the cell under test.  A scale factor controls
  how far above the noise mean a cell must be to be declared a
  detection.  This implementation is approximate and intended for
  experimentation rather than production use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter


@dataclass
class PercentileCFAR:
    """Simple global threshold CFAR.

    Attributes
    ----------
    percentile : float
        Percentile of the RD map used as the detection threshold.  A
        percentile of 99 corresponds to selecting the top 1% of cells.
    """

    percentile: float = 99.0

    def detect(self, rd_map: np.ndarray) -> np.ndarray:
        """Return a boolean mask where True indicates a detection.

        Parameters
        ----------
        rd_map : np.ndarray
            The input range–Doppler map (any real scale).

        Returns
        -------
        np.ndarray
            Boolean array of the same shape as `rd_map` with detected
            cells marked as True.
        """
        thresh = np.percentile(rd_map, self.percentile)
        return rd_map > thresh


@dataclass
class CACFAR:
    """Cell averaging CFAR implementation.

    Parameters
    ----------
    guard_cells : int
        Number of guard cells to exclude around the cell under test in
        both dimensions.  The guard region is a square of side
        `2*guard_cells+1`.
    ref_cells : int
        Number of reference cells on either side of the guard region.
        The complete reference window has side `2*(guard_cells+ref_cells)+1`.
    scale : float
        Factor by which the local noise estimate is multiplied to obtain
        the detection threshold.  A larger scale produces fewer detections.
    """

    guard_cells: int = 2
    ref_cells: int = 8
    scale: float = 3.0

    def detect(self, rd_map: np.ndarray) -> np.ndarray:
        """Return a detection mask using CA CFAR.

        Notes
        -----
        This implementation uses convolution via uniform filters to
        compute the sum of values within square windows.  It assumes
        periodic boundaries and does not handle edge effects carefully;
        therefore the returned mask should be considered an
        approximation.  For experimentation this is sufficient.
        """
        g = self.guard_cells
        r = self.ref_cells
        if r <= 0:
            raise ValueError("ref_cells must be positive")
        # total window size (guard + reference)
        win_size = 2 * (g + r) + 1
        guard_size = 2 * g + 1
        # compute sums over the large window and the guard window
        sum_all = uniform_filter(rd_map, size=win_size, mode="wrap") * (win_size ** 2)
        sum_guard = uniform_filter(rd_map, size=guard_size, mode="wrap") * (guard_size ** 2)
        # subtract guard window and current cell value to get sum of reference cells
        sum_ref = sum_all - sum_guard
        # number of reference cells
        num_ref = win_size ** 2 - guard_size ** 2
        # local noise estimate
        noise = sum_ref / num_ref
        threshold = noise * self.scale
        return rd_map > threshold


def cfar_from_config(cfg: dict) -> Tuple[object, str]:
    """Instantiate a CFAR detector from a configuration dictionary.

    The configuration must contain a `method` key.  Supported methods
    are `percentile` and `ca`.  Additional keys are passed to the
    corresponding detector class.

    Returns the detector instance and a descriptive name.
    """
    method = cfg.get("method", "percentile").lower()
    if method == "percentile":
        percentile = cfg.get("percentile", 99.0)
        return PercentileCFAR(percentile), f"Percentile-{percentile:.1f}"
    elif method in {"ca", "cacfar"}:
        guard = int(cfg.get("guard_cells", 2))
        ref = int(cfg.get("ref_cells", 8))
        scale = float(cfg.get("scale", 3.0))
        return CACFAR(guard, ref, scale), f"CA(g={guard},r={ref},s={scale:.1f})"
    else:
        raise ValueError(f"Unknown CFAR method: {method}")