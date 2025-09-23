# havok/havok_dae.py
"""
DAE-HAVOK module (library)
- DAE_HAVOK: Autoencoder + (learnable) Koopman K on latent
- make_dataloader: create sliding-window sequences
- train_dae_havok: train with reconstruction + 1-step Koopman loss
- encode_csv_with_dae / encode_array_with_dae: to latent sequences
- extract_K / compute_residual
- build_residual_havok: HAVOK on residuals r_t = z_{t+1} - K z_t
- build_latent_havok: HAVOK directly on latent z_t
- hierarchical_predict: simple K + residual HAVOK rollouts (optional)
- save_koopman_npz / load_koopman_npz, save_havok_npz / load_havok_npz
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.linalg import lstsq

from havok.build_havok import build_havok_model, load_csv_data

__all__ = [
    "DAE_HAVOK",
    "make_dataloader",
    "train_dae_havok",
    "encode_csv_with_dae",
    "encode_array_with_dae",
    "extract_K",
    "compute_residual",
    "build_residual_havok",
    "build_latent_havok",
    "hierarchical_predict",
    "save_koopman_npz",
    "load_koopman_npz",
    "save_havok_npz",
    "load_havok_npz",
]


# ------------------------- Model -------------------------


class DAE_HAVOK(nn.Module):
    """
    Deep Autoencoder + Koopman embedding (K) on latent space.

    Args
    ----
    input_dim : int
        Observed dimension D
    latent_dim : int
        Latent dimension r (K is r x r)

    Forward
    -------
    x_seq : (batch, seq_len, D)
    returns:
        z      : (batch, seq_len, r)
        z_pred : (batch, seq_len-1, r)     # 1-step via K
        x_rec  : (batch, seq_len, D)
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.r = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim)
        )
        self.K = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, input_dim)
        )

    def forward(
        self, x_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x_seq.shape
        z_flat = self.encoder(x_seq.view(-1, D))  # (B*L, r)
        z = z_flat.view(B, L, self.r)  # (B, L, r)
        z_past = z[:, :-1, :]  # (B, L-1, r)
        z_pred = torch.einsum("bij,jk->bik", z_past, self.K)
        x_rec = self.decoder(z_flat).view(B, L, D)
        return z, z_pred, x_rec


# ------------------------- Data -------------------------


def make_dataloader(
    ts: np.ndarray, seq_len: int, batch_size: int = 64, shuffle: bool = True
) -> DataLoader:
    """
    時系列 ts (T, D) を長さ seq_len のシーケンスに分割し DataLoader を返す。
    """
    ts = np.asarray(ts)
    if ts.ndim == 1:
        ts = ts[:, None]
    T, D = ts.shape
    if T <= seq_len:
        raise ValueError(f"seq_len ({seq_len}) must be < T ({T})")
    X, Y = [], []
    for i in range(T - seq_len):
        X.append(ts[i : i + seq_len])
        Y.append(ts[i + 1 : i + seq_len + 1])
    X = torch.tensor(np.stack(X), dtype=torch.float)
    Y = torch.tensor(np.stack(Y), dtype=torch.float)
    return DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=shuffle)


# ------------------------- Train -------------------------


def train_dae_havok(
    model: DAE_HAVOK,
    loader: DataLoader,
    lr: float = 1e-3,
    epochs: int = 50,
    alpha: float = 1.0,
) -> None:
    """
    Reconstruction loss + α * Koopman 1-step loss
    L = ||x_rec - x||^2 + α ||z_{t+1} - K z_t||^2
    """
    device = next(model.parameters()).device
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for ep in range(1, epochs + 1):
        total = 0.0
        model.train()
        for x_seq, _ in loader:
            x_seq = x_seq.to(device)
            opt.zero_grad()
            z, z_pred, x_rec = model(x_seq)
            L_rec = mse(x_rec, x_seq)
            L_dyn = mse(z_pred, z[:, 1:, :])
            loss = L_rec + alpha * L_dyn
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {ep}/{epochs} loss={total/len(loader):.4e}")


# ------------------------- Encode -------------------------


def _normalize_ts_like(ts: np.ndarray, D_expect: int) -> np.ndarray:
    ts = np.asarray(ts)
    if ts.ndim == 1:
        ts = ts[:, None]  # (T,) -> (T,1)
    if ts.ndim != 2:
        raise ValueError(f"array must be 1D or 2D, got shape {ts.shape}")
    if ts.shape[1] != D_expect:
        if ts.shape[0] == D_expect and ts.shape[1] != D_expect:
            ts = ts.T  # (1, T) など横持ち対応
        else:
            raise ValueError(f"feature dim {ts.shape[1]} != model.input_dim {D_expect}")
    return ts


def encode_array_with_dae(
    ts_np: np.ndarray, model: DAE_HAVOK, device: torch.device | None = None
) -> np.ndarray:
    """
    メモリ上の配列を潜在化。戻り値 (T, r)
    """
    if device is None:
        device = next(model.parameters()).device
    ts = _normalize_ts_like(ts_np, model.input_dim)
    model.eval().to(device)
    with torch.no_grad():
        z = model.encoder(torch.tensor(ts, dtype=torch.float, device=device))
    return z.detach().cpu().numpy()


def encode_csv_with_dae(
    filenames: list[str], model: DAE_HAVOK, device: torch.device | None = None
) -> np.ndarray:
    """
    CSV を読み込んで潜在化。内部で shape を自動正規化。
    """
    data_list = load_csv_data(filenames)
    ts = _normalize_ts_like(data_list[0], model.input_dim)
    if device is None:
        device = next(model.parameters()).device
    model.eval().to(device)
    with torch.no_grad():
        z = model.encoder(torch.tensor(ts, dtype=torch.float, device=device))
    return z.detach().cpu().numpy()


# ------------------------- Koopman & residual -------------------------


def extract_K(z_seq: np.ndarray, model: DAE_HAVOK | None = None) -> np.ndarray:
    """
    学習済み model.K があればそれ、なければ最小二乗で K を推定。
    """
    if model is not None and hasattr(model, "K"):
        return model.K.detach().cpu().numpy()
    Z = z_seq[:-1]
    Z_next = z_seq[1:]
    K_T, *_ = lstsq(Z, Z_next)  # Z K^T ≈ Z_next
    return K_T.T


def compute_residual(z_seq: np.ndarray, K: np.ndarray) -> np.ndarray:
    """r_t = z_{t+1} - K z_t   shape: (T-1, r)"""
    return z_seq[1:] - (z_seq[:-1] @ K.T)


# ------------------------- HAVOK wrappers -------------------------


def build_residual_havok(
    residual_seq: np.ndarray,
    dt: float = 1.0,
    delays: int = 50,
    rank: int | None = None,
    num_chaos: int = 1,
):
    """
    残差系列 r_t (T-1, r) に HAVOK を適用し、
    build_havok_model(data, dt, delays, rank, num_chaos) を呼ぶラッパ。
    戻り値: A, B, V_full, V_test, yout, u_test, t_test
    """
    data = residual_seq.T  # (r, T-1) = (channels, time)
    if rank is None:
        rank = data.shape[0]
    return build_havok_model(data, dt, delays, rank, num_chaos)


def build_latent_havok(
    z_seq: np.ndarray,
    dt: float = 1.0,
    delays: int = 50,
    rank: int | None = None,
    num_chaos: int = 1,
):
    """
    潜在系列 z_t (T, r) に HAVOK を直接適用するラッパ。
    戻り値: A, B, V_full, V_test, yout, u_test, t_test
    """
    data = z_seq.T  # (r, T)
    if rank is None:
        rank = data.shape[0]
    return build_havok_model(data, dt, delays, rank, num_chaos)


# ------------------------- Simple hierarchical predictor (optional) -------------------------


def hierarchical_predict(
    z0: np.ndarray, K: np.ndarray, A_r: np.ndarray, B_r: np.ndarray, steps: int
) -> np.ndarray:
    """
    簡易な階層予測（研究用途のラフ実装）。
    """
    r_dim = K.shape[0]
    r_t = np.zeros((r_dim,))
    z_t = z0.copy()
    preds = []
    for _ in range(steps):
        u_t = r_t[-1]
        r_next = A_r @ r_t[:-1] + B_r.flatten() * u_t
        z_t = K @ z_t + np.concatenate([r_next, [u_t]])[:r_dim]
        r_t = np.concatenate([r_next, [u_t]])
        preds.append(z_t.copy())
    return np.stack(preds)


# ------------------------- I/O helpers -------------------------


def save_koopman_npz(K: np.ndarray, path: str, meta: dict | None = None) -> str:
    """Koopman K を .npz 保存"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, K=K, meta=(meta or {}), saved_at=time.time())
    return path


def load_koopman_npz(path: str) -> tuple[np.ndarray, dict]:
    d = np.load(path, allow_pickle=True)
    K = d["K"]
    meta = d.get("meta", {}).item() if d.get("meta", None) is not None else {}
    return K, meta


def save_havok_npz(
    A: np.ndarray,
    B: np.ndarray,
    path: str,
    *,
    V_full: np.ndarray | None = None,
    V_test: np.ndarray | None = None,
    yout: np.ndarray | None = None,
    u_test: np.ndarray | None = None,
    t_test: np.ndarray | None = None,
    meta: dict | None = None,
) -> str:
    """HAVOK の A,B（＋任意で V,yout,u,t）を .npz 保存"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(A=A, B=B, meta=(meta or {}), saved_at=time.time())
    if V_full is not None:
        payload["V_full"] = V_full
    if V_test is not None:
        payload["V_test"] = V_test
    if yout is not None:
        payload["yout"] = yout
    if u_test is not None:
        payload["u_test"] = u_test
    if t_test is not None:
        payload["t_test"] = t_test
    np.savez_compressed(path, **payload)
    return path


def load_havok_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files if k not in ("meta",)}
    out["meta"] = d.get("meta", {}).item() if d.get("meta", None) is not None else {}
    return out
