
import numpy as np
from numpy.linalg import eigh

__all__ = [
    "ace_filter_2d",
    "eigen_clutter_suppression",
    "msd_per_bin_multichannel",
    "build_onehot_signature",
    "soft_clip",
]

def _extract_training_ring(x, r, c, half_win_r, half_win_c, guard_r, guard_c):
    """
    Collect training samples from a rectangular ring around (r,c).
    Returns an array of shape (num_train_pixels,).
    """
    R, C = x.shape
    r0 = max(0, r - half_win_r)
    r1 = min(R, r + half_win_r + 1)
    c0 = max(0, c - half_win_c)
    c1 = min(C, c + half_win_c + 1)

    # Guard region indices
    gr0 = max(0, r - guard_r)
    gr1 = min(R, r + guard_r + 1)
    gc0 = max(0, c - guard_c)
    gc1 = min(C, c + guard_c + 1)

    # Create mask for window minus guard box
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    mask[:] = True
    # Remove guard block within the window (adjusted coordinates)
    g_r0 = max(0, gr0 - r0)
    g_r1 = min(mask.shape[0], gr1 - r0)
    g_c0 = max(0, gc0 - c0)
    g_c1 = min(mask.shape[1], gc1 - c0)
    mask[g_r0:g_r1, g_c0:g_c1] = False

    # Training samples (complex)
    train = x[r0:r1, c0:c1][mask]
    return train

def ace_filter_2d(rd_map,
                  half_win=(6, 6),
                  guard=(2, 2),
                  patch=(3, 3),
                  epsilon=1e-3,
                  power="magnitude",
                  return_stat_only=True):
    """
    2D ACE (Adaptive Coherence Estimator) statistic over a Range–Doppler map.
    Treats each CUT as a delta-like target within a local patch; estimates
    a local covariance from a ring of training cells.

    Parameters
    ----------
    rd_map : (R, D) complex ndarray
        Complex Range–Doppler map (after FFT). Magnitude-only also accepted.
    half_win : (int, int)
        Half-size of the training window (range, doppler) around the CUT.
    guard : (int, int)
        Guard cells (range, doppler) to exclude around the CUT.
    patch : (int, int)
        Size of the local target template patch centered at the CUT.
        The target signature s is a one-hot vector at the patch center.
    epsilon : float
        Diagonal loading for covariance regularization.
    power : {"magnitude", "power"}
        How to precompute cell intensity for ACE; affects robustness.
    return_stat_only : bool
        If True, returns the ACE statistic map. If False, returns (stat, debug_dict).

    Returns
    -------
    ace_map : (R, D) float ndarray
        ACE statistic per bin in [0, 1]. Higher = more target-like.
    """
    x = rd_map.astype(np.complex128, copy=False)
    R, C = x.shape

    # Intensity to work with (ACE is scale-invariant; this choice is for stability)
    if power == "power":
        mag = np.real(x * np.conjugate(x))
    else:
        mag = np.abs(x)

    # Precompute indices for the patch vectorization
    ph, pw = patch
    assert ph % 2 == 1 and pw % 2 == 1, "patch must have odd dims (centered)"
    pr = ph // 2
    pc = pw // 2

    # Signature s is a one-hot at the patch center -> after whitening this becomes AMF/ACE
    s = np.zeros((ph * pw, 1), dtype=np.complex128)
    s[pr * pw + pc, 0] = 1.0

    ace = np.zeros_like(mag, dtype=np.float64)

    hr, hc = half_win
    gr, gc = guard

    for r in range(R):
        r0 = max(0, r - pr)
        r1 = min(R, r + pr + 1)
        for c in range(C):
            c0 = max(0, c - pc)
            c1 = min(C, c + pc + 1)

            # Extract local patch (pad by reflection if needed to keep patch size constant)
            patch_block = np.zeros((ph, pw), dtype=np.complex128)
            patch_block[pr - (r - r0):pr + (r1 - r),
                        pc - (c - c0):pc + (c1 - c)] = x[r0:r1, c0:c1]
            y = patch_block.reshape(-1, 1)  # (ph*pw, 1)

            # Training ring to estimate covariance (use magnitudes as weights)
            train = _extract_training_ring(x, r, c, hr, hc, gr, gc)
            if train.size < (ph * pw):
                # Not enough training; skip or use global estimate
                ace[r, c] = 0.0
                continue

            # Build sample covariance of vectorized patch pixels by assuming
            # independence across the patch directions but sharing a scalar variance
            # estimated from the ring (diagonal covariance). This is robust and cheap.
            sigma2 = np.mean(np.abs(train) ** 2)
            Rinv = (1.0 / (sigma2 + epsilon))

            # ACE statistic: (s^H R^{-1} y)^2 / [(s^H R^{-1} s)(y^H R^{-1} y)]
            num = np.abs((s.conj().T @ (Rinv * y)))[0, 0] ** 2
            den = ( (s.conj().T @ (Rinv * s))[0, 0].real *
                    ( (y.conj().T @ (Rinv * y))[0, 0].real ) + 1e-12 )
            ace[r, c] = float(num / den)

    if return_stat_only:
        return ace
    else:
        return ace, {"patch": patch, "half_win": half_win, "guard": guard}

def eigen_clutter_suppression(rd_map, half_win=(8, 8), guard=(2, 2), rank=1, epsilon=1e-3):
    """
    Local eigen-subspace canceller (simple and robust):
    - For each CUT, estimate a local mean magnitude from the ring.
    - Subtract it from the CUT magnitude (>=0). Acts like a local clutter remover.
    - (For stronger nulling, extend to project onto dominant eigenvectors of the local covariance.)

    Parameters
    ----------
    rd_map : (R, D) complex ndarray
    half_win : (int, int)
        Half-size of the training window.
    guard : (int, int)
        Guard cells excluded.
    rank : int
        Placeholder for future extension to true eigen-null; retained for API consistency.
    epsilon : float
        Unused in the simplified version; kept for API consistency.

    Returns
    -------
    residual_map : (R, D) float ndarray
    """
    x = rd_map.astype(np.complex128, copy=False)
    R, C = x.shape
    residual = np.zeros((R, C), dtype=np.float64)

    hr, hc = half_win
    gr, gc = guard

    for r in range(R):
        for c in range(C):
            ring = _extract_training_ring(x, r, c, hr, hc, gr, gc)
            if ring.size < 8:
                residual[r, c] = np.abs(x[r, c])
                continue
            residual[r, c] = max(0.0, np.abs(x[r, c]) - float(np.mean(np.abs(ring))))

    return residual

def msd_per_bin_multichannel(rd_cube, S, epsilon=1e-3):
    """
    Matched Subspace Detector per Range–Doppler bin for multi-channel data.

    Parameters
    ----------
    rd_cube : (R, D, C) complex ndarray
        Range–Doppler for C channels (e.g., antennas or virtual channels).
    S : (C, K) complex ndarray
        Signal subspace basis spanning expected target steering(s) across channels.
        Example: columns are array-manifold vectors at one or multiple angles.
    epsilon : float
        Diagonal loading for projection stability.

    Returns
    -------
    T : (R, D) float ndarray
        MSD statistic per bin: ||P_S y||^2 / (||P_{S^\\perp} y||^2 + 1e-12)
    """
    R, D, C = rd_cube.shape
    y = rd_cube.reshape(-1, C)  # (R*D, C)

    ShS = S.conj().T @ S + epsilon * np.eye(S.shape[1], dtype=np.complex128)
    P_S = S @ np.linalg.inv(ShS) @ S.conj().T  # (C, C)
    I = np.eye(C, dtype=np.complex128)
    P_N = I - P_S

    ys = (y @ P_S.conj().T)
    yn = (y @ P_N.conj().T)

    num = np.sum(np.abs(ys) ** 2, axis=1)
    den = np.sum(np.abs(yn) ** 2, axis=1) + 1e-12
    T = (num / den).reshape(R, D).astype(np.float64)
    return T

def build_onehot_signature(C, idx):
    """
    Convenience helper to create a one-hot subspace vector of length C (for MSD testing or AMF-like tests).
    """
    s = np.zeros((C, 1), dtype=np.complex128)
    s[idx % C, 0] = 1.0
    return s

def soft_clip(x, q=(1, 99)):
    """
    Percentile-based soft clipping to improve visualization.
    """
    lo, hi = np.percentile(x, q)
    return np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)
