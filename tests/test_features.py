import numpy as np

from radar_exp.features.rd_features import extract_rd_features
from radar_exp.features.pc_features import extract_pc_features
from radar_exp.detect.pointcloud import Point


def test_extract_rd_features() -> None:
    patch = np.array([[1.0, 2.0], [3.0, 4.0]])
    feats = extract_rd_features(patch)
    # Keys should exist
    assert set(feats.keys()) == {"mean", "var", "entropy", "centroid_r", "centroid_d", "peak_count"}
    # Mean should equal 2.5
    assert np.isclose(feats["mean"], 2.5)
    # Variance positive
    assert feats["var"] > 0
    # Peak count at least one
    assert feats["peak_count"] >= 1


def test_extract_pc_features() -> None:
    pts = [
        Point(0, 0, np.zeros(1), 1.0, 0.0, 0.0, 10.0, 1.0, 0.0),
        Point(0, 0, np.zeros(1), 2.0, 0.0, 0.0, 20.0, 2.0, 0.0),
    ]
    feats = extract_pc_features(pts)
    # Verify keys
    expected = {
        "centroid_x",
        "centroid_y",
        "range_mean",
        "range_std",
        "velocity_mean",
        "velocity_std",
        "doa_mean",
        "doa_std",
        "rcs_mean",
        "rcs_std",
        "point_count",
    }
    assert set(feats.keys()) == expected
    # Means
    assert np.isclose(feats["centroid_x"], 1.5)
    assert np.isclose(feats["range_mean"], 1.5)
    # Point count equals 2
    assert np.isclose(feats["point_count"], 2.0)