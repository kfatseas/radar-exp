import numpy as np

from radar_exp.detect.pointcloud import Point
from radar_exp.detect.patches import get_patch_bounds, extract_patch


def test_get_patch_bounds_and_extract() -> None:
    # Create a simple RD map and points
    rd_map = np.random.randn(30, 40)
    # Points at (range,doppler) = (5,5), (10,15), (20,30)
    points = [
        Point(r_idx=5, d_idx=5, snapshot=np.zeros(1), range_bin=0.0, velocity_bin=0.0,
              doa=0.0, rcs=0.0, x=0.0, y=0.0),
        Point(r_idx=10, d_idx=15, snapshot=np.zeros(1), range_bin=0.0, velocity_bin=0.0,
              doa=0.0, rcs=0.0, x=0.0, y=0.0),
        Point(r_idx=20, d_idx=30, snapshot=np.zeros(1), range_bin=0.0, velocity_bin=0.0,
              doa=0.0, rcs=0.0, x=0.0, y=0.0),
    ]
    bounds = get_patch_bounds(points, rd_map.shape, padding=2)
    r_min, r_max, d_min, d_max = bounds
    # Bounds should encompass all points plus padding
    assert r_min <= 5 - 2
    assert r_max >= 20 + 2 + 1
    assert d_min <= 5 - 2
    assert d_max >= 30 + 2 + 1
    patch = extract_patch(rd_map, bounds)
    # Patch shape matches bounds
    assert patch.shape == (r_max - r_min, d_max - d_min)