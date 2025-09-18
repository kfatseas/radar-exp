"""Patch extraction utilities.

Given a set of detection points belonging to an object, this module
computes the bounding box in the RD map, applies optional padding and
extracts the corresponding spectrum patch.  Patches are useful for
computing RD features and for visualisation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .pointcloud import Point


def get_patch_bounds(
    points: List[Point],
    rd_map_shape: Tuple[int, int],
    padding: int = 2,
) -> Tuple[int, int, int, int]:
    """Compute the bounding box (inclusive) for a set of points.

    Returns
    -------
    tuple
        `(r_min, r_max, d_min, d_max)` where the bounds are clamped to
        `[0, rd_map_shape[i] - 1]`.  The returned `r_max` and `d_max`
        are exclusive (suitable for slicing).
    """
    if not points:
        return (0, 0, 0, 0)
    r_idxs = [p.r_idx for p in points]
    d_idxs = [p.d_idx for p in points]
    r_min = max(0, min(r_idxs) - padding)
    r_max = min(rd_map_shape[0], max(r_idxs) + padding + 1)
    d_min = max(0, min(d_idxs) - padding)
    d_max = min(rd_map_shape[1], max(d_idxs) + padding + 1)
    return (r_min, r_max, d_min, d_max)


def extract_patch(
    rd_map: np.ndarray,
    bounds: Tuple[int, int, int, int],
) -> np.ndarray:
    """Extract a patch from the RD map given bounding box bounds.

    Parameters
    ----------
    rd_map : np.ndarray
        The complete RD map.
    bounds : tuple
        Bounding box `(r_min, r_max, d_min, d_max)` from `get_patch_bounds`.

    Returns
    -------
    np.ndarray
        The extracted patch (2‑D array).  No copying is performed; the
        returned array is a view into the original RD map.
    """
    r_min, r_max, d_min, d_max = bounds
    return rd_map[r_min:r_max, d_min:d_max]


def extract_patch_for_object(
    rd_map: np.ndarray,
    obj: List[Point],
    padding: int = 2,
) -> np.ndarray:
    """Convenience wrapper to extract a patch for an object.

    Parameters
    ----------
    rd_map : np.ndarray
        Range–Doppler map.
    obj : List[Point]
        Points belonging to an object.
    padding : int, optional
        Padding in RD bins around the object's bounding box.

    Returns
    -------
    np.ndarray
        A patch of the RD map containing the object and context.
    """
    bounds = get_patch_bounds(obj, rd_map.shape, padding)
    return extract_patch(rd_map, bounds)