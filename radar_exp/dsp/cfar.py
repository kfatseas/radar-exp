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


@dataclass
class ColumnNoiseCFAR:
    """Simple per-axis noise-floor CFAR.

    Computes a noise floor independently for each row (range) or
    column (Doppler) using a reduction statistic (median by default),
    then thresholds by adding a percentage margin.

    Attributes
    ----------
    axis : str
        Which axis to reduce over to compute the noise floor for each
        line. Use 'row'/'range' to compute one threshold per range row,
        or 'col'/'column'/'doppler' for one per Doppler column.
    percentage : float
        Threshold margin as a percentage of the noise floor. For
        example, 10.0 means threshold = noise * (1 + 0.10).
    reducer : str
        Statistic to compute the noise floor along the chosen axis.
        One of {'median', 'mean'}. Default is 'median' for robustness.
    clip_min : float | None
        Optional minimum noise floor to avoid extremely low thresholds.
    """

    axis: str = "row"
    percentage: float = 10.0
    reducer: str = "median"
    clip_min: float | None = None

    def detect(self, rd_map: np.ndarray) -> np.ndarray:
        if rd_map.ndim != 2:
            raise ValueError("rd_map must be a 2D array")
        # Determine reduction axis: for per-row thresholds, reduce across columns (axis=1);
        # for per-column thresholds, reduce across rows (axis=0).
        per_row = self.axis.lower() in {"row", "range"}
        reduce_axis = 1 if per_row else 0
        # Compute noise floor along the reduction axis
        if self.reducer.lower() == "median":
            noise = np.median(rd_map, axis=reduce_axis)
        elif self.reducer.lower() == "mean":
            noise = np.mean(rd_map, axis=reduce_axis)
        else:
            raise ValueError(f"Unknown reducer: {self.reducer}")
        # Optional clipping to avoid tiny thresholds
        if self.clip_min is not None:
            noise = np.maximum(noise, float(self.clip_min))
        # Expand to full map shape
        noise_map = noise[:, None] if per_row else noise[None, :]
        margin = 1.0 + (float(self.percentage) / 100.0)
        threshold = noise_map * margin
        return rd_map > threshold


def threshold_per_axis(
    rd_map: np.ndarray,
    axis: str = "col",
    percentage: float = 10.0,
    reducer: str = "median",
    clip_min: float | None = None,
) -> np.ndarray:
    """Threshold RD map using per-axis noise floor + percentage.

    Convenience function equivalent to using ColumnNoiseCFAR(...).detect(rd_map).

    Parameters
    ----------
    rd_map : np.ndarray
        2D range–Doppler map.
    axis : str
        'row'/'range' for one threshold per range row, or
        'col'/'column'/'doppler' for one per Doppler column.
    percentage : float
        Margin percentage added to the noise floor (e.g. 10.0 => +10%).
    reducer : str
        Reduction statistic along axis: 'median' or 'mean'.
    clip_min : float | None
        Optional minimum noise floor.

    Returns
    -------
    np.ndarray
        Boolean detection mask of same shape as rd_map.
    """
    det = ColumnNoiseCFAR(axis=axis, percentage=percentage, reducer=reducer, clip_min=clip_min)
    return det.detect(rd_map)


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
    elif method in {"column", "per_column", "row", "per_row", "axis"}:
        axis = cfg.get("axis", "row")
        percentage = float(cfg.get("percentage", 10.0))
        reducer = cfg.get("reducer", "median")
        clip_min = cfg.get("clip_min")
        det = ColumnNoiseCFAR(axis=axis, percentage=percentage, reducer=reducer, clip_min=clip_min)
        axis_name = "row" if axis.lower() in {"row", "range"} else "col"
        return det, f"Axis({axis_name},{reducer},+{percentage:.1f}%)"
    else:
        raise ValueError(f"Unknown CFAR method: {method}")