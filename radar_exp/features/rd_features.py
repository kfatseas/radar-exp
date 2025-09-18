"""Feature extraction from range–Doppler patches.

The functions in this module produce simple statistical summaries of
spectral patches.  These features are intentionally lightweight and
computationally inexpensive, making them suitable as baselines for
classification experiments.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
from scipy.stats import entropy as scipy_entropy


def extract_rd_features(patch: np.ndarray, resize: Optional[tuple] = None) -> Dict[str, float]:
    """Compute feature vector from an RD patch.

    Parameters
    ----------
    patch : np.ndarray
        Two‑dimensional array containing a contiguous region of an RD
        map.  Typically this patch has been cropped around an object.
    resize : tuple of int, optional
        If provided, the patch is resized to `(height, width)` using
        simple nearest neighbour interpolation before computing
        features.  This can normalise patches of varying sizes.  When
        `None` (default), the patch is used as is.

    Returns
    -------
    dict
        Dictionary of feature name to scalar value.  Keys include
        `mean`, `var`, `entropy`, `centroid_r`, `centroid_d` and
        `peak_count`.
    """
    # Optionally resize the patch
    data = patch.astype(float)
    if resize is not None:
        target_h, target_w = resize
        h, w = data.shape
        # Compute scale factors and sample grid
        row_indices = (np.linspace(0, h - 1, target_h)).astype(int)
        col_indices = (np.linspace(0, w - 1, target_w)).astype(int)
        data = data[np.ix_(row_indices, col_indices)]
    # Normalise patch to [0, 1] for entropy computation
    flat = data.flatten()
    # Avoid negative values by shifting
    min_val = float(flat.min())
    flat_shifted = flat - min_val
    total = float(flat_shifted.sum())
    if total > 0:
        probs = flat_shifted / total
        ent = float(scipy_entropy(probs))
    else:
        ent = 0.0
    mean = float(data.mean())
    var = float(data.var())
    # Weighted centroids along range (rows) and Doppler (cols)
    h, w = data.shape
    rows = np.arange(h)
    cols = np.arange(w)
    # Compute weights normalised to sum to one
    weight_sum = float(data.sum())
    if weight_sum > 0:
        w_rows = np.sum(data * rows[:, None]) / weight_sum
        w_cols = np.sum(data * cols[None, :]) / weight_sum
    else:
        w_rows = 0.0
        w_cols = 0.0
    # Peak count: number of local maxima above 90th percentile
    thresh = np.percentile(data, 90)
    peaks = (data > thresh)
    peak_count = int(np.count_nonzero(peaks))
    return {
        "mean": mean,
        "var": var,
        "entropy": ent,
        "centroid_r": float(w_rows),
        "centroid_d": float(w_cols),
        "peak_count": float(peak_count),
    }