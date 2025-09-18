"""Point cloud generation and object representation.

This module defines simple data structures for individual detection
points and clustered objects.  It includes utilities to estimate the
direction of arrival (DoA) from antenna snapshots and to convert
range–Doppler detections into physical point clouds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from scipy import signal
from sklearn.cluster import DBSCAN


@dataclass
class Point:
    """A single detection point.

    Attributes
    ----------
    r_idx : int
        Index on the range axis of the RD map.
    d_idx : int
        Index on the Doppler axis of the RD map.
    snapshot : np.ndarray
        Complex snapshot across antennas at this RD bin (shape `(antennas,)`).
    range_bin : float
        Physical range of the detection (metres).
    velocity_bin : float
        Physical radial velocity of the detection (m/s).  Positive values
        indicate approaching targets, negative values receding.
    doa : float
        Estimated direction of arrival in degrees.  Zero degrees
        corresponds to the radar boresight.
    rcs : float
        Radar cross section at the detection point (same scale as `rd_map`).
    x : float
        Cartesian x coordinate of the point (metres).
    y : float
        Cartesian y coordinate of the point (metres).
    """

    r_idx: int
    d_idx: int
    snapshot: np.ndarray
    range_bin: float
    velocity_bin: float
    doa: float
    rcs: float
    x: float
    y: float


@dataclass
class Object:
    """A clustered object consisting of multiple points.

    Attributes
    ----------
    label : int
        Cluster label assigned by DBSCAN.
    points : List[Point]
        List of points belonging to this object.
    patch : np.ndarray
        A binary mask of the RD map marking the object's detections.
    """

    label: int
    points: List[Point] = field(default_factory=list)
    rd_map_shape: Tuple[int, int] | None = None
    patch: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # Create a binary patch mask if rd_map_shape was provided
        if self.rd_map_shape is not None:
            patch = np.zeros(self.rd_map_shape, dtype=bool)
            for p in self.points:
                patch[p.r_idx, p.d_idx] = True
            self.patch = patch

    def __len__(self) -> int:
        return len(self.points)

    def centroid(self) -> Tuple[float, float]:
        """Return the centroid of the points in Cartesian coordinates."""
        if not self.points:
            return (0.0, 0.0)
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return (float(np.mean(xs)), float(np.mean(ys)))


def estimate_doa(snapshot: np.ndarray, fft_size: int = 181) -> float:
    """Estimate direction of arrival from an antenna snapshot.

    This function performs a 1‑D FFT along the antenna array and
    identifies the angle corresponding to the maximum magnitude bin.

    Parameters
    ----------
    snapshot : np.ndarray
        Complex snapshot of shape `(antennas,)`.
    fft_size : int, optional
        Number of points in the spatial FFT.  A larger size improves
        angular resolution.  Defaults to 181.

    Returns
    -------
    float
        Estimated DoA in degrees.
    """
    antennas = snapshot.shape[0]
    # Apply Chebyshev window to reduce sidelobes
    window = signal.windows.chebwin(antennas, at=80)
    windowed = snapshot * window
    # Zero pad to centre the antenna array within the FFT grid
    pad = fft_size // 2 - antennas // 2
    padded = np.pad(windowed, (pad, pad), mode="constant")
    spectrum = np.fft.fft(padded, n=fft_size)
    # Find the index of the maximum magnitude
    idx = int(np.argmax(np.abs(spectrum)))
    # Convert index to angle: map 0..N to -90..+90 degrees via arcsin
    # Normalise index to range [-1, 1]
    norm = 2.0 * idx / fft_size - 1.0
    norm = np.clip(norm, -1.0, 1.0)
    angle = np.degrees(np.arcsin(norm))
    return float(angle)


def create_points(
    rd_cube: np.ndarray,
    rd_map: np.ndarray,
    max_range: float,
    max_velocity: float,
    coords: np.ndarray,
    doa_fft_size: int = 181,
) -> List[Point]:
    """Construct a list of Point objects from RD detections.

    Parameters
    ----------
    rd_cube : np.ndarray
        The complex range–Doppler cube after FFTs (shape `(antennas, doppler_bins, range_bins)`).
    rd_map : np.ndarray
        The magnitude RD map used for thresholding (shape `(range_bins, doppler_bins)`).
    max_range : float
        Maximum detectable range in metres.
    max_velocity : float
        Maximum radial velocity in m/s.
    coords : np.ndarray
        Array of integer `(r_idx, d_idx)` coordinates indicating detections.
    doa_fft_size : int, optional
        FFT size for DoA estimation.  Defaults to 181.

    Returns
    -------
    List[Point]
        A list of populated `Point` instances.
    """
    points: List[Point] = []
    n_range, n_doppler = rd_map.shape
    # Precompute range and velocity bin centres
    range_bins = np.linspace(0.0, max_range, n_range)
    velocity_bins = np.linspace(-max_velocity, max_velocity, n_doppler)
    for (r_idx, d_idx) in coords:
        # rd_cube has shape (antennas, doppler, range)
        snapshot = rd_cube[:, d_idx, r_idx]
        rng = float(range_bins[r_idx])
        vel = float(velocity_bins[d_idx])
        doa = estimate_doa(snapshot, fft_size=doa_fft_size)
        rcs = float(rd_map[r_idx, d_idx])
        # Convert to Cartesian coordinates
        rad = np.radians(doa)
        x = rng * np.cos(rad)
        y = rng * np.sin(rad)
        pt = Point(r_idx, d_idx, snapshot, rng, vel, doa, rcs, x, y)
        points.append(pt)
    return points