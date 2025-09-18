"""Post‑processing utilities for detection masks.

After CFAR thresholding, detection masks may contain isolated
false alarms or fragmented blobs.  This module provides simple
morphological operations based on SciPy to clean up masks before
converting them into coordinate lists or patches.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing


def clean_mask(mask: np.ndarray, min_size: int = 1) -> np.ndarray:
    """Apply morphological opening and remove small connected components.

    Parameters
    ----------
    mask : np.ndarray
        Binary detection mask.
    min_size : int, optional
        Minimum number of pixels to keep.  Connected components smaller
        than this are removed.

    Returns
    -------
    np.ndarray
        Cleaned binary mask.
    """
    # opening removes single pixel speckle
    opened = binary_opening(mask)
    # closing fills small holes
    closed = binary_closing(opened)
    if min_size <= 1:
        return closed
    # label connected components to remove small ones
    # Use scipy.ndimage.label if available; otherwise keep as is
    try:
        from scipy.ndimage import label
        labeled, num = label(closed)
        out = np.zeros_like(mask, dtype=bool)
        for i in range(1, num + 1):
            component = (labeled == i)
            if np.sum(component) >= min_size:
                out |= component
        return out
    except Exception:
        return closed