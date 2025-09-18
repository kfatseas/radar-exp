import numpy as np

from radar_exp.dsp.fft import range_fft, velocity_fft


def test_range_velocity_fft_shapes() -> None:
    # Synthetic cube with antennas=4, chirps=8, samples=16
    cube = np.random.randn(4, 8, 16) + 1j * np.random.randn(4, 8, 16)
    range_cube = range_fft(cube)
    # rfft returns n_fft//2 + 1 bins
    assert range_cube.shape == (4, 8, 9)
    rd_cube = velocity_fft(range_cube)
    # The doppler dimension is fft_size (chirps) by default, equal to 8
    assert rd_cube.shape[0] == 4
    assert rd_cube.shape[1] == 8
    assert rd_cube.shape[2] == 9