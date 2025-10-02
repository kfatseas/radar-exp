"""Recording adapter to abstract dataset loading.

This module provides a thin wrapper around the user supplied
`src.recording.Recording` class.  It normalises access to radar cubes
and exposes useful properties such as the maximum range and velocity
defined in the dataset settings.

Users can subclass this class to integrate custom datasets, so long
as they implement the same public API (`get_frame`, `max_range`,
`max_velocity` and `n_frames`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RecordingAdapter:
    """Adapter around a radar recording.

    Parameters
    ----------
    path : str
        Path to the recording on disk.  The adapter will create an
        instance of `src.recording.Recording` internally.
    record_cls : Optional[type]
        Optionally override the recording class.  By default the
        constructor attempts to import `src.recording.Recording`.
    """

    path: str
    record_cls: Optional[type] = None

    def __post_init__(self) -> None:
        # Lazy import to avoid hard dependency when building documentation.
        if self.record_cls is None:
            try:
                from .dolphine import Recording  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "RecordingAdapter requires dolphine.Recording to be available"
                ) from exc
            self.record_cls = Recording
        # Instantiate underlying recording
        self.rec = self.record_cls(self.path)
        # Convert max velocity from km/h to m/s if needed
        if "max velocity" in self.rec.settings:
            v = float(self.rec.settings["max velocity"].item())
            self.rec.settings["max velocity"] = v / 3.6

    def get_frame(self, idx: int) -> Any:
        """Return the raw radar cube for the given frame index.

        The cube is expected to be a three dimensional numpy array of
        shape `(antennas, chirps, samples)`.
        """
        return self.rec.cube(idx)

    def max_range(self) -> float:
        """Return the maximum range of the radar (in metres)."""
        # Some recordings store numpy scalars; call item() to unwrap.
        return float(self.rec.settings["max distance"].item())

    def max_velocity(self) -> float:
        """Return the maximum unambiguous radial velocity (in m/s)."""
        return float(self.rec.settings["max velocity"].item())

    @property
    def n_frames(self) -> int:
        """Return the total number of frames in the recording."""
        # If __len__ is defined on the underlying recording, forward it.
        return len(self.rec)