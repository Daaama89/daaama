# havok/havok_HKAE.py
# -*- coding: utf-8 -*-
"""
HKAE + HAVOK-on-latent (uses your build_havok helpers if available)

Classes:
- HKAEConfig, HKAE, HKAETrainer
- HavokConfig, HavokLatentAnalyzer

Key equations (LaTeX):
- z_{k+1} = K z_k
- \mathcal{L}_\text{rec} = \frac{1}{N}\sum_i \lVert \hat{\mathbf h}_i - \mathbf h_i \rVert_2^2
- \mathcal{L}_\text{evo} = \frac{1}{H}\sum_{k=1}^{H}\frac{1}{N}\sum_i \lVert \hat{\mathbf h}_{i+k} - \mathbf h_{i+k}\rVert_2^2
- H = \begin{bmatrix} Z_{:,0:L}\\ Z_{:,1:L+1}\\ \dots \\ Z_{:,m-1:L+m-1} \end{bmatrix}
- H = U\Sigma V^\top,\ \ V\in\mathbb{R}^{r\times L}
- \dot V = A_\text{full} V,\ \ A=[A_\text{full}]_{1:r_c,1:r_c},\ B=[A_\text{full}]_{1:r_c,r_c+1:r}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import StateSpace, lsim
from scipy.signal import savgol_filter
from scipy.linalg import logm

# =========================
# Prefer user's helpers if present
# =========================
try:
    from .build_havok import (  # type: ignore
        compute_svd as _bh_compute_svd,
        differentiate as _bh_differentiate,
        compute_forcing_threshold as _bh_compute_thr,
    )
except Exception:
    _bh_compute_svd = None
    _bh_differentiate = None
    _bh_compute_thr = None


# =========================
# Local fallbacks
# =========================
def _np_svd(X: np.ndarray, svd_rank: int = 0):
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    rank = len(s) if svd_rank == 0 else min(svd_rank, len(s))
    return U[:, :rank], s[:rank], Vh[:rank, :]


def _central_diff(V: np.ndarray, dt: float):
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


def _diff_savgol(
    V: np.ndarray, dt: float, window: int = 11, poly: int = 3
) -> np.ndarray:
    """Smoothed derivative along time axis=1."""
    if window % 2 == 0:
        window += 1  # enforce odd
    return savgol_filter(
        V,
        window_length=window,
        polyorder=poly,
        deriv=1,
        delta=dt,
        axis=1,
        mode="interp",
    )


# Bind helpers (prefer user's)
def compute_svd(X: np.ndarray, svd_rank: int = 0):
    if _bh_compute_svd is not None:
        return _bh_compute_svd(X, svd_rank=svd_rank)
    return _np_svd(X, svd_rank=svd_rank)


def differentiate(V: np.ndarray, dt: float):
    if _bh_differentiate is not None:
        return _bh_differentiate(V, dt)
    return _central_diff(V, dt)


def compute_forcing_threshold(forcing: np.ndarray, percentile: int = 95):
    if _bh_compute_thr is not None:
        return _bh_compute_thr(forcing, percentile=percentile)
    return np.percentile(np.abs(forcing), percentile)


# =========================
# Delay stack
# =========================
def delay_stack(Z: np.ndarray, m: int, tau: int = 1) -> np.ndarray:
    """
    Build a vertical time-delay stack with row-centering.

    Parameters
    ----------
    Z : ndarray, shape (T, d)
        Time series in features.
    m : int
        Number of delay taps.
    tau : int
        Delay stride (in samples).

    Returns
    -------
    H : ndarray, shape (m*d, L)
        Row-centered Hankel-like delay matrix stacked vertically.

    Raises
    ------
    ValueError : if T - (m-1)*tau <= 0
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z must be 2D (T, d).")
    T, d = Z.shape
    L = T - (m - 1) * tau
    if L <= 0:
        raise ValueError("Sequence too short for the given (m, tau).")
    blocks = []
    for i in range(m):
        seg = Z[i * tau : i * tau + L, :].T  # (d, L)
        blocks.append(seg)
    H = np.vstack(blocks)  # (m*d, L)
    H = H - H.mean(axis=1, keepdims=True)  # center rows
    return H


# =========================
# Torch models (HKAE)
# =========================
class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class HKAE(nn.Module):
    def __init__(self, window_dim: int, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.encoder = Encoder(window_dim, latent_dim, hidden)
        self.decoder = Decoder(latent_dim, window_dim, hidden)
        # Koopman layer (no bias)
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, windows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        windows : (N, m)
            Each row is a delay-window sample.

        Returns
        -------
        z0 : (N, dz)
        x0_hat : (N, m)
        """
        z0 = self.encoder(windows)
        x0_hat = self.decoder(z0)
        return z0, x0_hat


# =========================
# Config + Trainer
# =========================
@dataclass
class HKAEConfig:
    m: int = 60
    tau: int = 1
    latent_dim: int = 8
    hidden: int = 128
    horizon: int = 2
    epochs: int = 20
    batch_size: int = 256
    lr: float = 2e-3
    device: str = "cpu"


def make_evolution_targets(H: np.ndarray, horizon: int = 1):
    """
    From (m, L) delay matrix, build training pairs for k-step evolution loss.

    Returns
    -------
    X : (N, m)
    targets : list of (N, m), length = horizon
    """
    m, L = H.shape
    N = L - horizon
    if N <= 0:
        raise ValueError("Not enough columns for the requested horizon.")
    X = H[:, :N].T
    targets = [H[:, k : k + N].T for k in range(1, horizon + 1)]
    return X, targets


class HKAETrainer:
    def __init__(self, cfg: HKAEConfig):
        self.cfg = cfg
        self.model: Optional[HKAE] = None

    def build_model(self, window_dim: int) -> HKAE:
        self.model = HKAE(
            window_dim=window_dim,
            latent_dim=self.cfg.latent_dim,
            hidden=self.cfg.hidden,
        ).to(self.cfg.device)
        return self.model

    def fit(self, H_train: np.ndarray, H_val: np.ndarray) -> HKAE:
        """
        Train HKAE with reconstruction + multi-step evolution loss.
        Keeps best validation checkpoint.
        """
        m, _ = H_train.shape
        model = self.build_model(window_dim=m)
        opt = optim.Adam(model.parameters(), lr=self.cfg.lr)
        mse = nn.MSELoss()

        Xtr, Ytr_list = make_evolution_targets(H_train, horizon=self.cfg.horizon)
        Xva, Yva_list = make_evolution_targets(H_val, horizon=self.cfg.horizon)
        Xtr = torch.tensor(Xtr, dtype=torch.float32, device=self.cfg.device)
        Xva = torch.tensor(Xva, dtype=torch.float32, device=self.cfg.device)
        Ytr_list = [
            torch.tensor(y, dtype=torch.float32, device=self.cfg.device)
            for y in Ytr_list
        ]
        Yva_list = [
            torch.tensor(y, dtype=torch.float32, device=self.cfg.device)
            for y in Yva_list
        ]

        def batching(X, Y_list, bs):
            n = X.shape[0]
            idx = torch.randperm(n)
            for i in range(0, n, bs):
                sel = idx[i : i + bs]
                yield X[sel], [Y[sel] for Y in Y_list]

        best_state = None
        best_val = np.inf

        for ep in range(1, self.cfg.epochs + 1):
            model.train()
            for xb, yb_list in batching(Xtr, Ytr_list, self.cfg.batch_size):
                opt.zero_grad()
                z0, x0_hat = model(xb)
                loss_rec = mse(x0_hat, xb)
                z = z0
                loss_evo = 0.0
                for yk in yb_list:
                    z = model.K(z)
                    xk_hat = model.decoder(z)
                    loss_evo += mse(xk_hat, yk)
                loss = loss_rec + loss_evo / len(yb_list)
                loss.backward()
                opt.step()

            # validation
            model.eval()
            with torch.no_grad():
                z0, x0_hat = model(Xva)
                loss_rec = mse(x0_hat, Xva).item()
                z = z0
                loss_evo = 0.0
                for yk in Yva_list:
                    z = model.K(z)
                    xk_hat = model.decoder(z)
                    loss_evo += mse(xk_hat, yk).item()
                va_loss = loss_rec + loss_evo / len(Yva_list)

            if va_loss < best_val:
                best_val = va_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

        if best_state is not None:
            model.load_state_dict(best_state)
        return model


# =========================
# HAVOK-on-latent (continuous-time)
# =========================
@dataclass
class HavokConfig:
    m_havok: int = 100
    tau: int = 1
    rank: int = 11
    horizon: int = 2
    num_chaos: int = 1
    dt: float = 0.01
    #  時間モデルの選択（"continuous" か "discrete"）
    time_model: str = "discrete"

    # #  連続時間まわりの安定化・微分・リッジ
    # diff_method: str = "savgol"  # "central" or "savgol"
    # savgol_window: int = 11  # odd
    # savgol_poly: int = 3
    ridge: float = 1e-3  # A推定のリッジ（連続/離散どちらでも使用可）
    # stabilize: bool = True  # 連続時間Aの左半平面シフト
    # stabilize_eps: float = 1e-6


class HavokLatentAnalyzer:
    def __init__(self, cfg: HavokConfig):
        self.cfg = cfg

    def latent_sequence(
        self, model: HKAE, H_full: np.ndarray, device: str = "cpu"
    ) -> np.ndarray:
        """
        Encode all windows (from H_full) into latent Z_t.
        """
        X_all, _ = make_evolution_targets(H_full, horizon=self.cfg.horizon)
        with torch.no_grad():
            X_all_t = torch.tensor(X_all, dtype=torch.float32, device=device)
            Z_all = model.encoder(X_all_t).cpu().numpy()  # (T', dz)
        return Z_all

    def build_delay_and_svd(self, Z_all: np.ndarray):
        """
        Build latent delay-stack and compute SVD (truncated).
        """
        HZ = delay_stack(Z_all, m=self.cfg.m_havok, tau=self.cfg.tau)  # (m*dz, L)
        U, s, V = compute_svd(HZ, svd_rank=self.cfg.rank)  # V: (r, L)
        return U, s, V, HZ

    def fit_continuous_havok(self, V: np.ndarray, s: np.ndarray):
        """
        Fit continuous-time HAVOK: dV/dt = A_full V, A = A_full[:r_c,:r_c], B = A_full[:r_c,r_c:].
        - 微分: central or Savitzky–Golay
        - 推定: Ridge 正則化
        - 安定化: 右半平面固有値を左にシフト（任意）
        """
        # 1) derivative
        if self.cfg.diff_method == "savgol":
            V_dot = _diff_savgol(
                V, self.cfg.dt, self.cfg.savgol_window, self.cfg.savgol_poly
            )
        else:
            V_dot = differentiate(V, self.cfg.dt)  # central difference fallback

        # 2) split
        L = V.shape[1]
        idx_train = int(0.8 * L)
        V_train = V[:, :idx_train]
        V_dot_train = V_dot[:, :idx_train]

        r = len(s)
        r_c = r - self.cfg.num_chaos
        if r_c <= 0:
            raise ValueError("rank - num_chaos must be >= 1")

        # 3) Ridge-regularized A_full (continuous)
        # A_full = (V_dot V^T)(V V^T + λI)^{-1}
        lam = float(self.cfg.ridge)
        G = V_train @ V_train.T
        A_full = (V_dot_train @ V_train.T) @ np.linalg.inv(G + lam * np.eye(G.shape[0]))

        A = A_full[:r_c, :r_c]
        B = A_full[:r_c, r_c:]

        # 4) stabilize (optional)
        if self.cfg.stabilize:
            w = np.linalg.eigvals(A)
            alpha = max(0.0, np.max(np.real(w)) + self.cfg.stabilize_eps)
            if alpha > 0:
                A = A - alpha * np.eye(A.shape[0])

        # 5) simulate (continuous) on test
        x0 = V[:r_c, idx_train]
        u_test = V[r_c:, idx_train:]
        sys = StateSpace(A, B, np.eye(r_c), np.zeros((r_c, self.cfg.num_chaos)))
        t_all = np.arange(L) * self.cfg.dt
        t_test = t_all[idx_train:]
        _, yout, _ = lsim(sys, u_test.T, t_test - t_test[0], x0)
        return A, B, V, u_test, t_test, yout

    def fit_discrete_havok(self, V: np.ndarray, s: np.ndarray):
        """
        Fit discrete-time HAVOK (DMDc-like):
        V_{k+1} = A_d_full V_k
        A_d = A_d_full[:r_c,:r_c], B_d = A_d_full[:r_c,r_c:]
        - 微分不要（ノイズに頑健）
        - Ridge 正則化
        """
        L = V.shape[1]
        idx_train = int(0.8 * L)

        r = len(s)
        r_c = r - self.cfg.num_chaos
        if r_c <= 0:
            raise ValueError("rank - num_chaos must be >= 1")

        # Train pairs
        V0 = V[:, : idx_train - 1]  # r x (N)
        V1 = V[:, 1:idx_train]  # r x (N)

        # Ridge-regularized least-squares for A_d_full
        # A_d_full = (V1 V0^T)(V0 V0^T + λI)^{-1}
        lam = float(self.cfg.ridge)
        G = V0 @ V0.T
        A_d_full = (V1 @ V0.T) @ np.linalg.inv(G + lam * np.eye(G.shape[0]))

        A_d = A_d_full[:r_c, :r_c]
        B_d = A_d_full[:r_c, r_c:]

        # test simulation
        x = V[:r_c, idx_train].copy()
        Utest = V[r_c:, idx_train:]  # (num_chaos, T_test)
        T_test = Utest.shape[1]
        Y = np.empty((T_test, r_c))
        for k in range(T_test):
            Y[k, :] = x
            uk = Utest[:, k]
            x = A_d @ x + B_d @ uk

        t_all = np.arange(L) * self.cfg.dt
        t_test = t_all[idx_train:]
        return A_d, B_d, V, Utest, t_test, Y

    def analyze(
        self, model: HKAE, H_full: np.ndarray, device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Full pipeline: HKAE encode -> latent delay SVD -> HAVOK fit (+ simulate).
        time_model:
        - "continuous": dV/dt = A V + B u（lsimで連続時間シミュレーション）
        - "discrete"  : V_{k+1} = A_d V_k + B_d u_k（逐次更新）
        """
        Z_all = self.latent_sequence(model, H_full, device=device)
        U, s, V, HZ = self.build_delay_and_svd(Z_all)

        if self.cfg.time_model == "discrete":
            A, B, V_full, u_test, t_test, yout = self.fit_discrete_havok(V, s)
        else:  # "continuous" (default)
            A, B, V_full, u_test, t_test, yout = self.fit_continuous_havok(V, s)

        return {
            "U": U,
            "s": s,
            "V": V_full,
            "HZ": HZ,
            "A": A,
            "B": B,
            "latent": Z_all,
            "u_test": u_test,
            "t_test": t_test,
            "yout": yout,
            "r": len(s),
        }
