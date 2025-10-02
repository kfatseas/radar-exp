import numpy as np

from radar_exp.detect.pointcloud import create_points


def test_create_points_accepts_mask() -> None:
    antennas = 4
    n_range = 2
    n_doppler = 3
    rd_cube = np.ones((antennas, n_doppler, n_range), dtype=np.complex64)
    rd_map = np.linspace(0.0, 1.0, n_range * n_doppler, dtype=np.float32).reshape(n_range, n_doppler)
    mask = np.zeros_like(rd_map, dtype=bool)
    mask[1, 2] = True

    points = create_points(rd_cube, rd_map, max_range=30.0, max_velocity=7.5, coords=mask)

    assert len(points) == 1
    pt = points[0]
    assert pt.r_idx == 1 and pt.d_idx == 2
    assert np.isclose(pt.range_bin, 30.0)
    assert np.isclose(pt.velocity_bin, 7.5)
    assert np.isclose(pt.x, pt.range_bin)
    assert np.isclose(pt.y, 0.0, atol=1e-6)
