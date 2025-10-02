"""Functions to build galleries of objects and their spectrum patches.

The gallery visualisation arranges multiple objects into a grid where
each column shows the RD spectrum patch alongside the corresponding
point cloud in Cartesian space.  This can be useful for qualitative
inspection of information loss.
"""

from __future__ import annotations

from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from ..detect.pointcloud import Object
from ..detect.patches import extract_patch


def gallery(
    rd_map: np.ndarray,
    objects: List[Object],
    padding: int = 2,
    max_cols: int = 4,
) -> None:
    """Display a gallery of objects.

    Parameters
    ----------
    rd_map : np.ndarray
        Complete RD map.
    objects : list of Object
        Objects to visualise.
    padding : int, optional
        Number of RD bins to pad around each object's bounding box.
    max_cols : int, optional
        Maximum number of columns in the grid.  Rows are added as
        necessary.
    """
    if not objects:
        print("No objects to visualise in gallery")
        return
    n = len(objects)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    for idx, obj in enumerate(objects):
        r = idx // cols
        c = idx % cols
        ax_pc, ax_spec = axes[r, c], axes[r, c]
        # Plot point cloud on top half
        xs = [p.x for p in obj.points]
        ys = [p.y for p in obj.points]
        ax = axes[r, c]
        ax.scatter(xs, ys, s=30, c='tab:blue', alpha=0.8)
        ax.set_title(f'Obj {obj.label} – Point Cloud')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        # Plot RD patch on bottom half (overlay second axes)
        # Create inset axes below the main axes
        bbox = ax.get_position()
        width = bbox.width
        height = bbox.height
        # Create new axes in figure coordinates
        inset = fig.add_axes([bbox.x0, bbox.y0 - height * 0.45, width, height * 0.4])
        # Extract patch
        from ..detect.patches import get_patch_bounds, extract_patch
        bounds = get_patch_bounds(obj.points, rd_map.shape, padding)
        patch = extract_patch(rd_map, bounds)
        inset.imshow(patch, origin='lower', aspect='auto', cmap='inferno')
        inset.set_title('RD Patch')
        inset.set_xlabel('Doppler bin')
        inset.set_ylabel('Range bin')
    plt.tight_layout()
    plt.show()