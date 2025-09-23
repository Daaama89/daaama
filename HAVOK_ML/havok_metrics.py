# havok/metrics.py
import numpy as np


def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def antisymmetry_report(A: np.ndarray) -> dict:
    F = lambda X: np.linalg.norm(X, "fro")
    S = 0.5 * (A + A.T)
    K = 0.5 * (A - A.T)
    eps_skew = F(A + A.T) / (F(A) + 1e-12)
    rho_K = F(K) / (F(A) + 1e-12)
    rho_S = F(S) / (F(A) + 1e-12)
    delta_diag = np.linalg.norm(np.diag(A)) / (F(A) + 1e-12)
    a = A.ravel()
    b = (-A.T).ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    lam = np.linalg.eigvals(A)
    eta_Re = float(
        np.linalg.norm(np.real(lam)) / (np.linalg.norm(np.imag(lam)) + 1e-12)
    )
    return {
        "eps_skew(||A+A^T||_F/||A||_F)": float(eps_skew),
        "rho_K(||K||_F/||A||_F)": float(rho_K),
        "rho_S(||S||_F/||A||_F)": float(rho_S),
        "delta_diag(||diag(A)||_2/||A||_F)": float(delta_diag),
        "corr(vec(A), vec(-A^T))": float(corr),
        "eig_ratio_eta_Re(||Re(λ)||/||Im(λ)||)": float(eta_Re),
        "spectral_radius": spectral_radius(A),
    }


def koopman_residual_stats(z: np.ndarray, K: np.ndarray) -> dict:
    """z: (T,r), K: (r,r) → r_t=z_{t+1}-K z_t の統計"""
    R = z[1:] - z[:-1] @ K.T
    l2 = np.linalg.norm(R, axis=1)
    return {
        "residual_mean_l2": float(l2.mean()),
        "residual_std_l2": float(l2.std(ddof=1)) if l2.size > 1 else 0.0,
        "residual_max_l2": float(l2.max()),
    }
