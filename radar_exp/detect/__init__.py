"""Detection and point cloud generation.

This package contains functions for converting range–Doppler maps into
lists of detection points, clustering those points into objects and
extracting RD patches.
"""

from . import peaks, cluster, patches, pointcloud

__all__ = ["peaks", "cluster", "patches", "pointcloud"]