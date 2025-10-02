"""Peak detection from range–Doppler maps.

This module glues together CFAR detectors with simple post‑processing to
produce lists of detection coordinates.  The main entry point
`detect_peaks` accepts a range–Doppler map and configuration
dictionaries for CFAR and post‑processing.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..dsp.cfar import cfar_from_config
from ..dsp.postproc import clean_mask


def detect_peaks(
    rd_map: np.ndarray,
    cfar_cfg: Dict[str, object] | None = None,
    postproc_cfg: Dict[str, object] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect peaks in a range–Doppler map and return their coordinates.

    Parameters
    ----------
    rd_map : np.ndarray
        Input RD map (real valued).  Peaks correspond to cells of
        interest.
    cfar_cfg : dict, optional
        Configuration for the CFAR detector.  See
        `radar_exp.dsp.cfar.cfar_from_config` for details.  If `None`
        defaults to a 99th percentile threshold.
    postproc_cfg : dict, optional
        Configuration for post‑processing (e.g., minimum component
        size).  Keys: `min_size`.

    Returns
    -------
    coords : np.ndarray
        Array of shape `(n_detections, 2)` with (row, column) indices
        into `rd_map` for each detection.
    mask : np.ndarray
        Boolean mask of shape equal to `rd_map` marking detection cells.
    """
    if cfar_cfg is None:
        cfar_cfg = {"method": "percentile", "percentile": 99.0}
    if postproc_cfg is None:
        postproc_cfg = {}
    # Instantiate detector
    det, _ = cfar_from_config(cfar_cfg)
    method = cfar_cfg.get('method', 'percentile').lower()
    if method in {'ca', 'cacfar', 'axis'}:
        # CA CFAR and axis CFAR require linear scale for proper calculations
        linear_map = 10**(rd_map / 20)
        raw_mask = det.detect(linear_map)
    else:
        raw_mask = det.detect(rd_map)
    # Clean mask
    min_size = int(postproc_cfg.get("min_size", 1))
    mask = clean_mask(raw_mask, min_size=min_size)
    # Convert to coordinates
    coords = np.column_stack(np.where(mask))
    return coords, mask