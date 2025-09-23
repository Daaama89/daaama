"""
Module: build_havok
Provides functions to load data, compute SVD, differentiate,
and build/run a HAVOK model.
"""

import numpy as np
from scipy.signal import StateSpace, lsim


def load_csv_data(filenames, delimiter=",", dtype=float, encoding="utf-8"):
    """
    Load one or more CSV files into a list of numpy arrays.

    Parameters
    ----------
    filenames : list of str
        Paths to CSV files.
    delimiter : str
    dtype : data-type
    encoding : str

    Returns
    -------
    data_list : list of ndarray
    """
    return [
        np.genfromtxt(f, delimiter=delimiter, dtype=dtype, encoding=encoding)
        for f in filenames
    ]


def compute_svd(X, svd_rank=0):
    """
    Perform SVD on X and truncate to `svd_rank` modes.
    Returns U, singular values s, and V matrix (modes x time).
    """
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    rank = len(s) if svd_rank == 0 else min(svd_rank, len(s))
    return U[:, :rank], s[:rank], Vh[:rank, :]


def differentiate(V, dt):
    """
    Compute time derivative of each row in V using central differences.

    Parameters
    ----------
    V : ndarray, shape (modes, time)
    dt : float

    Returns
    -------
    V_dot : ndarray, same shape as V
    """
    V = np.asarray(V)
    if V.ndim == 1:
        V = V[None, :]
    V_dot = np.empty_like(V)
    # central differences
    V_dot[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2 * dt)
    # forward/backward
    V_dot[:, 0] = (-3 * V[:, 0] + 4 * V[:, 1] - V[:, 2]) / (2 * dt)
    V_dot[:, -1] = (3 * V[:, -1] - 4 * V[:, -2] + V[:, -3]) / (2 * dt)
    return V_dot


def compute_forcing_threshold(forcing, percentile=95):
    """
    Determine a threshold for forcing activation based on percentile of |forcing|.
    """
    return np.percentile(np.abs(forcing.flatten()), percentile)


def build_havok_model(data, dt, delays, rank, num_chaos):
    """
    Construct and simulate the HAVOK linear model from input timeseries data.

    Parameters
    ----------
    data : ndarray, shape (channels, time)
        Multivariate timeseries (e.g., stacked x,y,z).
    dt : float
        Sampling interval for analysis.
    delays : int
        Number of delay coordinates.
    rank : int
        Number of modes to retain after SVD.
    num_chaos : int
        Number of forcing modes (typically 1).

    Returns
    -------
    A : ndarray, shape (r-num_chaos, r-num_chaos)
    B : ndarray, shape (r-num_chaos, num_chaos)
    V_full : ndarray, shape (r, N)
    V_test : ndarray, shape (r, N_test)
    yout : ndarray, shape (N_test, r-num_chaos)
    u_test : ndarray, shape (num_chaos, N_test)
    t_test : ndarray, shape (N_test,)
    """
    # 1) Build Hankel matrix
    _, m = data.shape
    Hm = m - (delays - 1)
    H = np.vstack([data[:, i : i + Hm] for i in range(delays)])
    # center
    H -= H.mean(axis=1, keepdims=True)

    # 2) SVD
    U, s, V = compute_svd(H, svd_rank=rank)
    V_dot = differentiate(V, dt)

    # 3) Train/test split
    N = V.shape[1]
    idx_train = int(0.8 * N)
    idx_test = int(0.9 * N)
    V_train = V[:, :idx_train]
    V_dot_train = V_dot[:, :idx_train]

    # 4) Estimate A, B
    A_full = np.linalg.lstsq(V_train.T, V_dot_train.T, rcond=None)[0].T
    r = len(s)
    A = A_full[: r - num_chaos, : r - num_chaos]
    B = A_full[: r - num_chaos, r - num_chaos :]

    # 5) Simulate on test forcing
    x0 = V[: r - num_chaos, idx_train]
    u_test = V[r - num_chaos :, idx_train:idx_test]
    sys = StateSpace(A, B, np.eye(r - num_chaos), np.zeros((r - num_chaos, num_chaos)))
    t_all = np.arange(N) * dt
    t_test = t_all[idx_train:idx_test]
    _, yout, _ = lsim(sys, u_test.T, t_test - t_test[0], x0)

    V_full = V
    V_test = V[:, idx_train:idx_test]
    return A, B, V_full, V_test, yout, u_test, t_test
