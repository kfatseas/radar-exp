"""Feature extraction from point clouds.

Point cloud features summarise the spatial distribution and signal
strength of clustered detections.  These features are intentionally
simple to serve as baselines for comparing against RD patch features.
"""

from __future__ import annotations

from typing import List, Dict

import numpy as np

from ..detect.pointcloud import Point


def extract_pc_features(points: List[Point]) -> Dict[str, float]:
    """Compute feature vector from a list of points.

    Parameters
    ----------
    points : list of Point
        Points belonging to a single object.

    Returns
    -------
    dict
        Mapping feature name to scalar value.  Keys include
        `centroid_x`, `centroid_y`, `range_mean`, `range_std`,
        `velocity_mean`, `velocity_std`, `doa_mean`, `doa_std`,
        `rcs_mean`, `rcs_std` and `point_count`.
    """
    if not points:
        # Return zeros when no points are present
        return {
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "range_mean": 0.0,
            "range_std": 0.0,
            "velocity_mean": 0.0,
            "velocity_std": 0.0,
            "doa_mean": 0.0,
            "doa_std": 0.0,
            "rcs_mean": 0.0,
            "rcs_std": 0.0,
            "point_count": 0.0,
        }
    # Convert to numpy arrays
    xs = np.array([p.x for p in points], dtype=float)
    ys = np.array([p.y for p in points], dtype=float)
    ranges = np.array([p.range_bin for p in points], dtype=float)
    velocities = np.array([p.velocity_bin for p in points], dtype=float)
    doas = np.array([p.doa for p in points], dtype=float)
    rcss = np.array([p.rcs for p in points], dtype=float)
    # Compute features
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    range_mean = float(ranges.mean())
    range_std = float(ranges.std())
    velocity_mean = float(velocities.mean())
    velocity_std = float(velocities.std())
    doa_mean = float(doas.mean())
    doa_std = float(doas.std())
    rcs_mean = float(rcss.mean())
    rcs_std = float(rcss.std())
    point_count = float(len(points))
    return {
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "range_mean": range_mean,
        "range_std": range_std,
        "velocity_mean": velocity_mean,
        "velocity_std": velocity_std,
        "doa_mean": doa_mean,
        "doa_std": doa_std,
        "rcs_mean": rcs_mean,
        "rcs_std": rcs_std,
        "point_count": point_count,
    }