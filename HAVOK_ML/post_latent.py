# havok/post_latent.py
# -*- coding: utf-8 -*-
"""
Back-projection from latent HAVOK to the original observation space,
and diagnostics/metrics for validity checking.

Required objects from your pipeline:
- HKAE model (with .encoder/.decoder)
- SVD of latent-delay stack: U, s
- HAVOK result dict: res = {"A","B","U","s","V","yout","u_test","t_test","r",...}
- HKAE Hankel config: m (original window size), and latent-delay m_havok

References:
- scipy.linalg.logm/expm (discrete<->continuous), SciPy docs.
- scipy.signal.StateSpace/lsim (continuous simulation), SciPy docs.
- numpy.linalg.svd (SVD), NumPy docs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

from scipy.linalg import logm, expm  # matrix log/exp

# (StateSpace/lsim は連続時間検証で使うときのみ)
from scipy.signal import StateSpace, lsim

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from scipy.signal import welch  # PSD 指標に使用


# ---------- 基本ユーティリティ ----------


def discrete_to_continuous(
    A_d: np.ndarray, B_d: np.ndarray, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    x_{k+1} = A_d x_k + B_d u_k  ->  d/dt x = A x + B u
    via block matrix logarithm:
        exp(dt * [[A, B],[0, 0]]) = [[A_d, B_d],[0, I]]
    """
    n, m = A_d.shape[0], B_d.shape[1]
    Phi_d = np.block([[A_d, B_d], [np.zeros((m, n)), np.eye(m)]])
    Phi_c = (1.0 / dt) * logm(Phi_d)
    A_c = np.real_if_close(Phi_c[:n, :n])
    B_c = np.real_if_close(Phi_c[:n, n:])
    return A_c, B_c


def symmetric_skew_ratio(A: np.ndarray) -> float:
    """
    ||Sym(A)||_F / ||Skew(A)||_F ; 反対称性の度合い（小さいほど反対称に近い）
    """
    S = 0.5 * (A + A.T)  # symmetric
    K = 0.5 * (A - A.T)  # skew
    num = np.linalg.norm(S, ord="fro")
    den = np.linalg.norm(K, ord="fro") + 1e-12
    return float(num / den)


def kurtosis_excess(x: np.ndarray) -> float:
    """
    超過尖度（ガウス=0）。母数版の簡易実装。
    """
    x = np.asarray(x).ravel()
    mu = x.mean()
    s2 = x.var()
    if s2 <= 0:
        return 0.0
    m4 = np.mean((x - mu) ** 4)
    return float(m4 / (s2**2) - 3.0)


def compute_active_threshold(vr: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(np.abs(vr.ravel()), percentile))


def build_Vhat_from_latent_havok(res: Dict[str, np.ndarray]) -> np.ndarray:
    """
    res["yout"] : (T_test, r_c), res["u_test"] : (num_chaos, T_test)
    -> V_hat : (r, T_test)
    """
    yout = res["yout"]  # (T_test, r_c)
    u_test = res["u_test"]  # (num_chaos, T_test)
    r = int(res["r"])
    r_c = yout.shape[1]
    assert r_c + u_test.shape[0] == r, "r mismatch with r_c+num_chaos"
    V_hat = np.vstack([yout.T, u_test])  # (r, T_test)
    return V_hat


# ---------- “遅延スタック”の逆変換（斜め平均; SSAのHankel化） ----------


def diagonal_average_stacked(H_stack: np.ndarray, m: int, d: int) -> np.ndarray:
    """
    H_stack: (m*d, L)  where each column = [ z_t, z_{t+1}, ..., z_{t+m-1} ] stacked (size m*d)
    return sequence Z_hat: (T_hat, d),  with T_hat = L + m - 1
    Reconstruction by anti-diagonal (Hankel) averaging.
    """
    H = H_stack.reshape(m, d, -1)  # (m, d, L)
    L = H.shape[2]
    T_hat = L + m - 1
    Z = np.zeros((T_hat, d))
    W = np.zeros((T_hat, 1))  # number of contributions per time index

    # column k contributes to times k..k+m-1
    for k in range(L):
        for i in range(m):
            t = k + i
            Z[t, :] += H[i, :, k]
            W[t, 0] += 1.0

    Z /= W + 1e-12
    return Z


# ---------- 潜在HAVOK -> 観測空間の時系列復元 ----------


@dataclass
class BackProjectConfig:
    m_orig: int  # HKAE の元ウィンドウ長 (args.m)
    m_havok: int  # 潜在側の遅延段数
    dt: float
    center_pick: bool = (
        True  # 復元時にウィンドウの中心をサンプリング（端より位相が安定）
    )


def predict_observation_from_latent(
    model,  # HKAE
    res: Dict[str, np.ndarray],
    U: np.ndarray,
    s: np.ndarray,
    cfg: BackProjectConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    1) V_hat (r x T_test) を UΣ で latent-delay stack HZ_hat に戻す
    2) 斜め平均で Z_hat (Tz x dz) を復元
    3) 各 z_t を decoder に通し、長さ m_orig の観測ウィンドウを得る
    4) 観測ウィンドウを再び斜め平均して x_pred を得る

    Returns
    -------
    t_pred : (N,)
    x_pred : (N,)
    stats  : {"len_Z":..., "len_HZ":..., "len_x":...}
    """
    # --- 1) latent-delay stack の再構成 ---
    V_hat = build_Vhat_from_latent_havok(res)  # (r, T_test)
    Sig = np.diag(s)  # (r, r)
    HZ_hat = (U @ Sig) @ V_hat  # (m_havok*dz, T_test)

    # --- 2) 斜め平均で Z_t 復元 ---
    # dz は encoder 出力次元
    dz = (
        model.encoder.net[-1].out_features
        if hasattr(model.encoder.net[-1], "out_features")
        else model.decoder.net[0].in_features
    )
    Z_hat = diagonal_average_stacked(HZ_hat, m=cfg.m_havok, d=dz)  # (Tz, dz)

    # --- 3) decoder で観測ウィンドウへ ---
    # バッチで高速化
    import torch

    with torch.no_grad():
        z_t = torch.tensor(Z_hat, dtype=torch.float32)
        windows = model.decoder(z_t).cpu().numpy()  # (Tz, m_orig)

    # --- 4) 斜め平均で 1次元時系列へ ---
    m = cfg.m_orig
    L = windows.shape[0]
    N = L + m - 1
    x = np.zeros((N, 1))
    w = np.zeros((N, 1))
    for k in range(L):
        for i in range(m):
            t = k + i
            x[t, 0] += windows[k, i]
            w[t, 0] += 1.0
    x /= w + 1e-12
    x_pred = x.ravel()

    # 時刻軸（res["t_test"][0] を起点に復元長に合わせる）
    t0 = float(res["t_test"][0])
    t_pred = t0 + cfg.dt * np.arange(N)

    stats = {
        "len_Vhat": V_hat.shape[1],
        "len_HZ": HZ_hat.shape[1],
        "len_Z": Z_hat.shape[0],
        "len_x": N,
    }
    return t_pred, x_pred, stats


# ---------- 妥当性指標（latent/係数/復元観測） ----------


def metrics_latent_havok(
    res: Dict[str, np.ndarray],
    A_d: np.ndarray,
    B_d: np.ndarray,
    dt: float,
    V_full: np.ndarray,  # (r, L)  真値の遅延SVD座標
    x_ref: Optional[np.ndarray] = None,
    t_ref: Optional[np.ndarray] = None,
    x_pred: Optional[np.ndarray] = None,
    t_pred: Optional[np.ndarray] = None,
    percentile: float = 95.0,
) -> Dict[str, float]:
    """
    代表的な妥当性指標を辞書で返す。
    """
    out: Dict[str, float] = {}
    # 基本
    yout = res["yout"]
    u_test = res["u_test"]
    t_test = res["t_test"]
    r = int(res["r"])
    r_c = yout.shape[1]

    # 1) latentの再構成誤差（テスト区間）
    idx_train = int(0.8 * V_full.shape[1])
    Tt = yout.shape[0]
    V_true = V_full[:r_c, idx_train : idx_train + Tt].T  # (Tt, r_c)
    rmse_modes = np.sqrt(np.mean((V_true - yout) ** 2, axis=0))
    out["rmse_v1"] = float(rmse_modes[0])
    out["rmse_modes_mean"] = float(rmse_modes.mean())

    # 2) 強制項の間欠度
    vr = u_test[0, :]
    thr = compute_active_threshold(vr, percentile=percentile)
    active_frac = float(np.mean(np.abs(vr) > thr))
    out["forcing_active_fraction@p{}".format(int(percentile))] = active_frac
    out["forcing_kurtosis"] = kurtosis_excess(vr)

    # 3) 離散->連続への変換性（行列対数）
    A_c, B_c = discrete_to_continuous(A_d, B_d, dt)
    out["cont_skew_ratio"] = symmetric_skew_ratio(A_c)
    eigA = np.linalg.eigvals(A_c)
    out["cont_eig_real_max"] = float(np.max(np.real(eigA)))

    # 4) 行列の往復誤差（d->c->d）
    n, m = A_d.shape[0], B_d.shape[1]
    Phi_c = np.block([[A_c, B_c], [np.zeros((m, n)), np.zeros((m, m))]])
    Phi_d_hat = expm(dt * Phi_c)
    A_d_hat = Phi_d_hat[:n, :n]
    B_d_hat = Phi_d_hat[:n, n:]
    out["roundtrip_err_A"] = float(
        np.linalg.norm(A_d - A_d_hat, "fro") / (np.linalg.norm(A_d, "fro") + 1e-12)
    )
    out["roundtrip_err_B"] = float(
        np.linalg.norm(B_d - B_d_hat, "fro") / (np.linalg.norm(B_d, "fro") + 1e-12)
    )

    # 5) 観測空間の一致度 (あれば)
    if x_ref is not None and x_pred is not None:
        # 時刻合わせ（単純内挿で共通区間 RMSE）
        t0 = max(t_ref[0], t_pred[0])
        t1 = min(t_ref[-1], t_pred[-1])
        if t1 > t0:
            dt0 = dt
            grid = np.arange(t0, t1 + 1e-12, dt0)
            # 線形補間
            xr = np.interp(grid, t_ref, x_ref)
            xp = np.interp(grid, t_pred, x_pred)
            out["rmse_obs"] = float(np.sqrt(np.mean((xr - xp) ** 2)))
            # 相関係数
            xr_c = (xr - xr.mean()) / (xr.std() + 1e-12)
            xp_c = (xp - xp.mean()) / (xp.std() + 1e-12)
            out["corr_obs"] = float(np.mean(xr_c * xp_c))
    return out


def _build_uniform_grid(t: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
    """
    一様グリッドを構成する補助関数。
    dt が未指定のときは t の中央値差を用いる。
    """
    t = np.asarray(t).ravel()
    if dt is None:
        diffs = np.diff(t)
        dt = float(np.median(diffs))
    N = int(np.floor((t[-1] - t[0]) / dt)) + 1
    grid = t[0] + dt * np.arange(N)
    return grid, dt


def compare_timeseries_and_plot(
    t_ref: np.ndarray,
    x_ref: np.ndarray,
    t_pred: np.ndarray,
    x_pred: np.ndarray,
    dt: Optional[float] = None,
    out_dir: Optional[str] = None,
    filename_base: str = "obs_compare",
    active_intervals: Optional[List[Tuple[float, float]]] = None,
    title: str = "Original vs Predicted (observation)",
) -> Dict[str, np.ndarray]:
    """
    ① 元の時系列と学習予測の比較（整列・残差・可視化・保存（任意））

    手順（数式）：
    - 共通一様グリッド 𝒯 を作成し、線形補間で
      \[
        x_r(t)\to x_r(\mathcal{T}),\quad x_p(t)\to x_p(\mathcal{T})
      \]
      を得る。
    - 残差 \(e = x_p(\mathcal{T}) - x_r(\mathcal{T})\) を計算。
    - 上段：オーバーレイ、下段：残差。active_intervals があれば赤帯で強調。

    Parameters
    ----------
    t_ref, x_ref : 元系列の時刻と値（1次元）
    t_pred, x_pred : 予測系列の時刻と値（1次元）
    dt : 共通グリッド刻み（省略時は t_ref の中央値差）
    out_dir : 保存先。指定時、PNG と npy（整列済みデータと残差）を保存
    filename_base : 出力ファイル名の接頭辞
    active_intervals : [(t_start, t_end), ...] を指定すると赤帯で強制活性区間を強調
    title : 図タイトル

    Returns
    -------
    out : dict
        {
          "t_grid": ..., "x_ref_grid": ..., "x_pred_grid": ..., "residual": ...
        }
    """
    # 共通グリッド作成
    grid, dt_eff = _build_uniform_grid(t_ref, dt)

    # 有効区間（両系列が定義される交差）に制限
    t0 = max(grid[0], float(t_pred[0]))
    t1 = min(grid[-1], float(t_pred[-1]))
    mask = (grid >= t0) & (grid <= t1)
    grid = grid[mask]

    # 線形補間で整列（1D）
    xr = np.interp(grid, t_ref, x_ref)  # numpy.interp を使用
    xp = np.interp(grid, t_pred, x_pred)

    # 残差
    res = xp - xr

    # 可視化
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    axs[0].plot(grid, xr, label="original", lw=1.0)
    axs[0].plot(grid, xp, "--", label="predicted", lw=1.0)
    axs[0].set_ylabel("value")
    axs[0].set_title(title)
    axs[0].legend(loc="upper right")

    axs[1].plot(grid, res, lw=0.9)
    axs[1].axhline(0.0, color="k", lw=0.7, alpha=0.5)
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("residual")

    # 強制活性区間の赤帯
    if active_intervals:
        for ts, te in active_intervals:
            axs[0].axvspan(ts, te, color="red", alpha=0.15)
            axs[1].axvspan(ts, te, color="red", alpha=0.15)

    fig.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        png_path = os.path.join(out_dir, f"{filename_base}.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        # 整列結果と残差を npy 保存（pickle不要の純配列）
        np.save(os.path.join(out_dir, f"{filename_base}_t_grid.npy"), grid)
        np.save(os.path.join(out_dir, f"{filename_base}_x_ref.npy"), xr)
        np.save(os.path.join(out_dir, f"{filename_base}_x_pred.npy"), xp)
        np.save(os.path.join(out_dir, f"{filename_base}_residual.npy"), res)
    else:
        plt.show()

    return {"t_grid": grid, "x_ref_grid": xr, "x_pred_grid": xp, "residual": res}


def show_learning_metrics(
    x_ref_grid: np.ndarray,
    x_pred_grid: np.ndarray,
    dt: float,
    out_dir: Optional[str] = None,
    filename: str = "metrics_observation",
) -> Dict[str, float]:
    """
    ② 学習の成功度合いを示す指標を表示（print）し、npy 保存（任意）

    指標（観測空間の一致度）：
    - RMSE：\(\sqrt{\frac{1}{N}\sum (x_p-x_r)^2}\)
    - MAE ：\(\frac{1}{N}\sum |x_p-x_r|\)
    - NRMSE（標準偏差基準）：\(\frac{\mathrm{RMSE}}{\sigma(x_r)}\)
    - 相関係数（Pearson）
    - 決定係数 \(R^2 = 1 - \frac{\sum (x_p-x_r)^2}{\sum (x_r-\bar{x}_r)^2}\)
    - スペクトルRMSE（WelchのPSDで比較；同一 fs=1/dt）
    - 位相ずれ（最大相互相関のラグ秒）

    Parameters
    ----------
    x_ref_grid, x_pred_grid : 同じ時間グリッド上に整列済みの 1D ベクトル
    dt : サンプリング周期（秒）
    out_dir, filename : 指標の .npy を保存したい場合に指定

    Returns
    -------
    metrics : dict（数値指標）
    """
    xr = np.asarray(x_ref_grid).ravel()
    xp = np.asarray(x_pred_grid).ravel()
    assert xr.shape == xp.shape, "整列後ベクトルの長さが一致していません。"

    N = xr.size
    err = xp - xr

    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    std = float(np.std(xr) + 1e-12)
    nrmse = float(rmse / std)

    # 相関係数
    xr_c = (xr - xr.mean()) / (xr.std() + 1e-12)
    xp_c = (xp - xp.mean()) / (xp.std() + 1e-12)
    corr = float(np.mean(xr_c * xp_c))

    # R^2
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((xr - xr.mean()) ** 2) + 1e-12)
    r2 = float(1.0 - ss_res / ss_tot)

    # PSD（Welch）でのスペクトルRMSE
    fs = 1.0 / float(dt)
    f_r, Pxx_r = welch(xr, fs=fs, nperseg=min(1024, N))
    f_p, Pxx_p = welch(xp, fs=fs, nperseg=min(1024, N))
    # 周波数軸を参照側に合わせて内挿
    Pxx_p_i = np.interp(f_r, f_p, Pxx_p)
    psd_rmse = float(np.sqrt(np.mean((Pxx_p_i - Pxx_r) ** 2)))

    # 交差相関でラグ推定（サンプル単位 → 秒）
    xc = np.correlate(xr_c, xp_c, mode="full")
    lags = np.arange(-N + 1, N)
    k = int(np.argmax(xc))
    lag_samples = int(lags[k])
    lag_sec = float(lag_samples * dt)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "NRMSE_std": nrmse,
        "Corr": corr,
        "R2": r2,
        "PSD_RMSE": psd_rmse,
        "Lag_sec_at_max_xcorr": lag_sec,
        "N": float(N),
        "dt": float(dt),
    }

    # 表示（丸めて見やすく）
    pretty = {
        k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()
    }
    print("[metrics/observation]", pretty)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{filename}.npy"), metrics, allow_pickle=True)

    return metrics


def compare_timeseries_and_plot_segmented(
    t_ref: np.ndarray,
    x_ref: np.ndarray,
    t_pred: np.ndarray,
    x_pred: np.ndarray,
    n_segments: int = 5,
    dt: Optional[float] = None,
    out_dir: Optional[str] = None,
    filename_base: str = "obs",
    active_intervals: Optional[List[Tuple[float, float]]] = None,
    title_prefix: str = "Original vs Predicted",
) -> Dict[str, Any]:
    """
    元の時系列と予測の比較図（重ね描き＋残差）を時間で n_segments 分割して描画/保存する。

    手順:
      1) 共通一様グリッドに線形補間して整列（x_ref_grid, x_pred_grid）。
      2) 残差 res = x_pred_grid - x_ref_grid を計算。
      3) grid の時間範囲を等分割して、各区間ごとに 2 段（上:重ね描き / 下:残差）を描画。

    保存:
      - 画像: {filename_base}_seg{ii:02d}.png
      - 各セグメントの npy:
          {filename_base}_seg{ii:02d}_t.npy
          {filename_base}_seg{ii:02d}_x_ref.npy
          {filename_base}_seg{ii:02d}_x_pred.npy
          {filename_base}_seg{ii:02d}_residual.npy

    Returns
    -------
    {
      "t_grid": ･･･全区間のグリッド,
      "x_ref_grid": ･･･,
      "x_pred_grid": ･･･,
      "residual": ･･･,
      "segments": [
         {"range": (t0, t1), "t": ..., "x_ref": ..., "x_pred": ..., "residual": ...},
         ...
      ]
    }
    """
    # --- まずは既存ロジックで整列 ---
    grid, dt_eff = _build_uniform_grid(t_ref, dt)

    # 有効区間（両系列の共通）に制限
    t0_all = max(grid[0], float(t_pred[0]))
    t1_all = min(grid[-1], float(t_pred[-1]))
    mask_all = (grid >= t0_all) & (grid <= t1_all)
    grid = grid[mask_all]

    xr = np.interp(grid, t_ref, x_ref)
    xp = np.interp(grid, t_pred, x_pred)
    res = xp - xr

    # --- セグメント境界 ---
    if n_segments < 1:
        n_segments = 1
    edges = np.linspace(grid[0], grid[-1], n_segments + 1)

    results = {
        "t_grid": grid,
        "x_ref_grid": xr,
        "x_pred_grid": xp,
        "residual": res,
        "segments": [],
    }

    # 出力先
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # --- 各セグメントを描画・保存 ---
    for ii in range(n_segments):
        t_start, t_end = edges[ii], edges[ii + 1]
        seg_mask = (grid >= t_start) & (
            grid < t_end if ii < n_segments - 1 else grid <= t_end
        )
        if not np.any(seg_mask):
            continue

        t_seg = grid[seg_mask]
        xr_seg = xr[seg_mask]
        xp_seg = xp[seg_mask]
        res_seg = res[seg_mask]

        fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        # 上段：重ね描き
        axs[0].plot(t_seg, xr_seg, label="original", lw=1.0)
        axs[0].plot(t_seg, xp_seg, "--", label="predicted", lw=1.0)
        axs[0].set_ylabel("value")
        axs[0].set_title(
            f"{title_prefix}  (segment {ii+1}/{n_segments}: {t_start:.3f}–{t_end:.3f})"
        )
        axs[0].legend(loc="upper right")

        # 下段：残差
        axs[1].plot(t_seg, res_seg, lw=0.9)
        axs[1].axhline(0.0, color="k", lw=0.7, alpha=0.5)
        axs[1].set_xlabel("time")
        axs[1].set_ylabel("residual")

        # 強制活性区間の赤帯（区間に交差している部分のみ描画）
        if active_intervals:
            for ts, te in active_intervals:
                # 可視領域との交差区間をとる
                a = max(ts, t_start)
                b = min(te, t_end)
                if b > a:
                    axs[0].axvspan(a, b, color="red", alpha=0.15)
                    axs[1].axvspan(a, b, color="red", alpha=0.15)

        fig.tight_layout()

        if out_dir is not None:
            png_path = os.path.join(out_dir, f"{filename_base}_seg{ii:02d}.png")
            fig.savefig(png_path, dpi=150)
            plt.close(fig)

            # 各セグメントの整列結果を npy 保存
            np.save(os.path.join(out_dir, f"{filename_base}_seg{ii:02d}_t.npy"), t_seg)
            np.save(
                os.path.join(out_dir, f"{filename_base}_seg{ii:02d}_x_ref.npy"), xr_seg
            )
            np.save(
                os.path.join(out_dir, f"{filename_base}_seg{ii:02d}_x_pred.npy"), xp_seg
            )
            np.save(
                os.path.join(out_dir, f"{filename_base}_seg{ii:02d}_residual.npy"),
                res_seg,
            )
        else:
            plt.show()

        results["segments"].append(
            {
                "range": (float(t_start), float(t_end)),
                "t": t_seg,
                "x_ref": xr_seg,
                "x_pred": xp_seg,
                "residual": res_seg,
            }
        )

    return results
