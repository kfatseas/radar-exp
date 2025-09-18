"""Clustering of detection points into objects.

This module groups individual detections into objects using DBSCAN.  The
clustering is performed in the space of physical attributes (range,
velocity and angle).  The resulting objects carry the original points
and an RD patch mask.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from .pointcloud import Point, Object


def cluster_points(
    points: List[Point],
    rd_map_shape: Tuple[int, int],
    eps: float = 1.0,
    min_samples: int = 5,
) -> List[Object]:
    """Cluster points into objects using DBSCAN.

    Parameters
    ----------
    points : List[Point]
        The list of points to cluster.
    rd_map_shape : tuple of int
        Shape of the RD map `(range_bins, doppler_bins)`.  Needed to
        initialise patch masks in the returned objects.
    eps : float, optional
        Maximum distance between two samples for one to be considered
        as in the neighbourhood of the other.  Defaults to 1.0.
    min_samples : int, optional
        Minimum number of points in a neighbourhood for a sample to be
        considered a core point.  Defaults to 5.

    Returns
    -------
    List[Object]
        A list of clustered objects.  Noise points are discarded.
    """
    if not points:
        return []
    # Build feature array: [range, velocity, doa]
    feats = np.array([[p.range_bin, p.velocity_bin, p.doa] for p in points])
    # Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(feats)
    objects = {}
    for point, label in zip(points, labels):
        if label == -1:
            # noise
            continue
        objects.setdefault(label, []).append(point)
    # Create Object instances
    return [Object(label, pts, rd_map_shape) for label, pts in objects.items()]