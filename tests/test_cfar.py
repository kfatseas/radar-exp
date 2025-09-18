import numpy as np

from radar_exp.dsp.cfar import PercentileCFAR, CACFAR


def test_percentile_cfar_detects_strong_cell() -> None:
    rd_map = np.zeros((10, 10))
    rd_map[5, 5] = 10.0
    detector = PercentileCFAR(percentile=90.0)
    mask = detector.detect(rd_map)
    # The strong cell should be detected
    assert mask[5, 5]
    # The rest should be mostly False
    assert np.sum(mask) == 1


def test_ca_cfar_reference_cells() -> None:
    rd_map = np.zeros((20, 20))
    rd_map[10, 10] = 10.0
    detector = CACFAR(guard_cells=1, ref_cells=2, scale=2.0)
    mask = detector.detect(rd_map)
    # Should detect the strong cell
    assert mask[10, 10]