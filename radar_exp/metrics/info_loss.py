"""Information‑loss metrics between RD patches and point clouds.

The purpose of these functions is to quantify how much of the
information present in a spectral patch is retained when converted
into a sparse point cloud.  Several simple metrics are provided:

* **Energy retention**: the fraction of total spectral energy captured
  by the detections.
* **Sparsity**: ratio of number of detections to total number of pixels
  in the patch.
* **Entropy difference**: difference between the Shannon entropy of the
  RD patch and that of the RCS values of the corresponding points.
"""

from __future__ import annotations

from typing import List, Dict

import numpy as np

from ..detect.pointcloud import Point


def energy_retention(patch: np.ndarray, points: List[Point]) -> float:
    """Compute the fraction of RD patch energy captured by points.

    The patch is assumed to contain intensities in dB.  It is
    converted to linear magnitude before integration.  The energy
    captured by the points is the sum of the linear magnitudes at
    their RD coordinates.
    """
    # Convert dB to linear
    linear = 10.0 ** (patch / 20.0)
    total_energy = float(np.sum(linear))
    if total_energy <= 0:
        return 0.0
    # Sum energies of detected bins within this patch
    captured = 0.0
    # Determine patch origin to offset indices
    # We assume that the points come from this patch; therefore subtract
    # patch origin when indexing.  In this simple implementation we
    # ignore offset and rely on exact matching of r_idx and d_idx.
    for p in points:
        try:
            captured += float(linear[p.r_idx, p.d_idx])
        except Exception:
            # Point outside patch
            continue
    return captured / total_energy


def sparsity(patch: np.ndarray, points: List[Point]) -> float:
    """Compute the density of points within the patch.

    Defined as the ratio of the number of points to the total number
    of pixels in the patch.  Lower density implies more aggressive
    pruning and hence greater information loss.
    """
    h, w = patch.shape
    n_points = len(points)
    return float(n_points) / float(h * w) if h * w > 0 else 0.0


def entropy_difference(patch: np.ndarray, points: List[Point]) -> float:
    """Compute the difference in entropy between patch and point RCS.

    Both patch and point RCS values are converted to probability
    distributions via normalisation.  The difference of their Shannon
    entropies is returned.  A large positive value indicates that the
    point cloud distribution is more concentrated (lower entropy) than
    the full patch.
    """
    # Flatten patch and shift to nonnegative
    flat = patch.flatten().astype(float)
    flat_shift = flat - float(flat.min())
    total = float(flat_shift.sum())
    if total > 0:
        p_dist = flat_shift / total
        ent_patch = -float(np.sum(p_dist * np.log2(p_dist + 1e-12)))
    else:
        ent_patch = 0.0
    # Point RCS distribution
    if points:
        rcs_vals = np.array([p.rcs for p in points], dtype=float)
        rcs_shift = rcs_vals - float(rcs_vals.min())
        rcs_sum = float(rcs_shift.sum())
        if rcs_sum > 0:
            q = rcs_shift / rcs_sum
            ent_points = -float(np.sum(q * np.log2(q + 1e-12)))
        else:
            ent_points = 0.0
    else:
        ent_points = 0.0
    return ent_patch - ent_points


def compute_info_loss_metrics(patch: np.ndarray, points: List[Point]) -> Dict[str, float]:
    """Compute all information‑loss metrics for a patch and its points."""
    return {
        "energy_retention": energy_retention(patch, points),
        "sparsity": sparsity(patch, points),
        "entropy_diff": entropy_difference(patch, points),
    }