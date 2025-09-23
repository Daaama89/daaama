# havok/havok_dae.py
"""
DAE-HAVOK module (library only)
- DAE_HAVOK: End-to-end Autoencoder + Koopman embedding (K) + HAVOK residual dynamics (on residuals)
- make_dataloader: Sequence dataset builder
- train_dae_havok: Training loop with reconstruction + Koopman losses
- encode_csv_with_dae: CSVファイルから潜在系列を生成
- extract_K: 潜在線形作用素 K を抽出（モデル内 or 最小二乗）
- compute_residual: r_t = z_{t+1} - K z_t を計算
- build_residual_havok: 残差系列に対して HAVOK 同定（ハンケル→SVD→線形）
- hierarchical_predict: K + residual-HAVOK を用いた階層予測（latent）

※ 実行用スクリプトは `scripts/run_dae_havok.py` を参照してください（本ファイルには含めません）。
"""
from __future__ import annotations
import time, json, os


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
    "hierarchical_predict",
]


class DAE_HAVOK(nn.Module):
    """
    Deep Autoencoder + Koopman embedding (K) + residual-HAVOK on latent space.

    Args:
        input_dim: 観測次元 D
        latent_dim: 潜在次元 r（K は r×r）

    Forward Inputs:
        x_seq: (batch, seq_len, D)

    Forward Returns:
        z: (batch, seq_len, r)                  # 潜在系列
        z_pred: (batch, seq_len-1, r)           # K による 1-step 予測
        x_rec: (batch, seq_len, D)              # 復元
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.r = latent_dim
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim)
        )
        # Koopman operator K（学習可能パラメータ）
        self.K = nn.Parameter(torch.randn(latent_dim, latent_dim))
        # Decoder network for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, input_dim)
        )

    def forward(
        self, x_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x_seq.shape
        # Encode each time step independently
        z_flat = self.encoder(x_seq.view(-1, D))  # (B*L, r)
        z = z_flat.view(B, L, self.r)  # (B, L, r)
        # 1-step linear prediction via K
        z_past = z[:, :-1, :]  # (B, L-1, r)
        z_pred = torch.einsum("bij,jk->bik", z_past, self.K)  # (B, L-1, r)
        # Reconstruction
        x_rec = self.decoder(z_flat).view(B, L, D)
        return z, z_pred, x_rec


def make_dataloader(
    ts: np.ndarray, seq_len: int, batch_size: int = 64, shuffle: bool = True
) -> DataLoader:
    """
    時系列 ts を長さ seq_len のシーケンスに分割し DataLoader を返す。

    Args:
        ts: (T, D)
        seq_len: シーケンス長
        batch_size: バッチサイズ
        shuffle: シャッフル有無

    Returns:
        DataLoader: (x_seq, y_seq) を返すローダ（y_seq は未使用でもつ）
    """
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


def train_dae_havok(
    model: DAE_HAVOK,
    loader: DataLoader,
    lr: float = 1e-3,
    epochs: int = 50,
    alpha: float = 1.0,
) -> None:
    """
    DAE_HAVOK の学習（復元 MSE + α・Koopman 1-step MSE）。

    Args:
        model: DAE_HAVOK インスタンス
        loader: (x_seq, y_seq) を返す DataLoader
        lr: 学習率
        epochs: エポック数
        alpha: Koopman 損失の重み
    """
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for ep in range(1, epochs + 1):
        total = 0.0
        model.train()
        for x_seq, _ in loader:
            x_seq = x_seq.to(device)
            optimizer.zero_grad()
            z, z_pred, x_rec = model(x_seq)
            L_rec = mse(x_rec, x_seq)
            L_dyn = mse(z_pred, z[:, 1:, :])
            loss = L_rec + alpha * L_dyn
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {ep}/{epochs}  loss={total/len(loader):.4f}")


def encode_csv_with_dae(
    filenames: list[str],
    model: DAE_HAVOK,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    CSV を読み込み、エンコーダで一括潜在化。

    Returns:
        latent_seq: (T, r)
    """
    data_list = load_csv_data(filenames)
    ts = data_list[0]
    if device is None:
        device = next(model.parameters()).device
    model.eval().to(device)
    with torch.no_grad():
        ts_tensor = torch.tensor(ts, dtype=torch.float, device=device)
        z = model.encoder(ts_tensor)
    return z.detach().cpu().numpy()


def encode_array_with_dae(
    ts_np: np.ndarray,
    model: DAE_HAVOK,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    メモリ上の配列 ts_np をエンコードして潜在系列を返す（CSV 再読込なし）。

    入力:
      ts_np: (T, D) または (T,) または (D, T) の実数配列
    戻り値:
      latent_seq: (T, r)
    """
    ts = np.asarray(ts_np)

    # 形状正規化: (T,) -> (T,1)
    if ts.ndim == 1:
        ts = ts[:, None]
    if ts.ndim != 2:
        raise ValueError(f"array must be 1D or 2D, got shape {ts.shape}")

    D_expect = model.input_dim
    # 列数が一致しなければ転置で揃えられるか確認
    if ts.shape[1] != D_expect:
        if ts.shape[0] == D_expect and ts.shape[1] != D_expect:
            ts = ts.T
        else:
            raise ValueError(
                f"array feature dim {ts.shape[1]} != model.input_dim {D_expect}"
            )

    if device is None:
        device = next(model.parameters()).device
    model.eval().to(device)
    with torch.no_grad():
        z = model.encoder(torch.tensor(ts, dtype=torch.float, device=device))
    return z.detach().cpu().numpy()


def extract_K(z_seq: np.ndarray, model: DAE_HAVOK | None = None) -> np.ndarray:
    """
    潜在系列から Koopman 演算子 K を取得。
    - model に学習済み K があればそれを返す
    - 無ければ最小二乗で K を推定（Z Z^+）
    """
    if model is not None and hasattr(model, "K"):
        return model.K.detach().cpu().numpy()
    Z = z_seq[:-1]
    Z_next = z_seq[1:]
    # 最小二乗: Z K^T ≈ Z_next  => K^T = Z^+ Z_next
    K_T, *_ = lstsq(Z, Z_next)
    return K_T.T


def compute_residual(z_seq: np.ndarray, K: np.ndarray) -> np.ndarray:
    """r_t = z_{t+1} - K z_t を計算（shape: (T-1, r)）。"""
    return z_seq[1:] - (z_seq[:-1] @ K.T)


# 置き換え：build_residual_havok
def build_residual_havok(
    residual_seq: np.ndarray,
    dt: float = 1.0,
    delays: int = 100,
    rank: int | None = None,
    num_chaos: int = 1,
):
    """
    残差系列に対して HAVOK 同定を実行（build_havok_model(data, dt, delays, rank, num_chaos) に合わせたラッパ）。

    Parameters
    ----------
    residual_seq : ndarray, shape (T-1, r)
        潜在の残差系列（時間×特徴）
    dt : float
        サンプリング間隔
    delays : int
        遅れ座標数（ハンケルの行方向）
    rank : int | None
        SVD保持モード数（Noneなら自動設定）
    num_chaos : int
        強制項（外力）モード数（通常 1）

    Returns
    -------
    A, B, V_full, V_test, yout, u_test, t_test
        build_havok_model と同じ戻り値
    """
    # build_havok_model は data を (channels, time) で受け取るので転置
    # residual_seq: (T-1, r) → data: (r, T-1)
    data = residual_seq.T

    # rank を未指定なら安全側で自動設定
    # ※ SVD はハンケル行列 H のランクに依存しますが、ここでは上限を channels にしておくのが無難
    if rank is None:
        rank = data.shape[0]  # = r

    # 本家シグネチャに合わせて位置引数で渡す
    return build_havok_model(data, dt, delays, rank, num_chaos)


def hierarchical_predict(
    z0: np.ndarray,
    K: np.ndarray,
    A_r: np.ndarray,
    B_r: np.ndarray,
    steps: int,
) -> np.ndarray:
    """
    階層モデル予測（簡易版）:
      1) residual の HAVOK から r_t を一次近似（u_t は r の最終成分を使用する簡略化）
      2) z_{t+1} = K z_t + r_t

    Note:
      実運用では residual 系の状態定義を厳密化し、外力項の取り扱いを明示することを推奨。
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


# --- 追加: モデル成果物の保存/読込ユーティリティ -------------------------
def save_koopman_npz(K: np.ndarray, path: str, meta: dict | None = None) -> str:
    """Koopman 作用素 K を .npz で保存"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(K=K, meta=(meta or {}), saved_at=time.time())
    np.savez_compressed(path, **payload)
    return path


def load_koopman_npz(path: str) -> tuple[np.ndarray, dict]:
    """K を読み込み"""
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
    """HAVOK の A,B（必要に応じて V, yout, u_test, t_test も）を .npz 保存"""
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
    """HAVOK成果物を辞書で読み込み（A,B,他）"""
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files if k not in ("meta",)}
    out["meta"] = d.get("meta", {}).item() if d.get("meta", None) is not None else {}
    return out


# -------------------------------------------------------------------------
