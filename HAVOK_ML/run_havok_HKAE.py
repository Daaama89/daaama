# scripts/run_havok_HKAE.py
# -*- coding: utf-8 -*-
"""
Runner for HKAE + HAVOK-on-latent (uses your build_havok helpers if present)
Plots are updated to 'active-slice (red highlight)' style you provided.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import expm

from havok.havok_HKAE import (
    HKAEConfig,
    HKAETrainer,
    HavokConfig,
    HavokLatentAnalyzer,
    delay_stack,
)
from havok.post_latent import (
    BackProjectConfig,
    predict_observation_from_latent,
    metrics_latent_havok,
    discrete_to_continuous,
    compare_timeseries_and_plot,
    show_learning_metrics,
    compare_timeseries_and_plot_segmented,
)

# =========================
# Optional helpers from your modules
# =========================
# A/B 可視化（存在すれば優先利用）
try:
    from havok.plotting import plot_matrix_and_vector_with_shared_colorbar as plot_AB
except Exception:
    plot_AB = None

# 閾値（存在すれば優先利用）
try:
    from havok.build_havok import compute_forcing_threshold as _compute_thr
except Exception:
    _compute_thr = None

# build_havok のローダを優先的に利用
try:
    from havok.build_havok import load_csv_data as bh_load_csv_data
except Exception:
    bh_load_csv_data = None


def compute_forcing_threshold(forcing, percentile=95):
    """|forcing| のパーセンタイルで対称しきい値。"""
    if _compute_thr is not None:
        return _compute_thr(forcing, percentile=percentile)
    return np.percentile(np.abs(forcing), percentile)


# =========================
# Fallback / local helpers (プロット関数群)
# =========================
def plot_matrix_and_vector_with_shared_colorbar_fallback(A, B, save_path=None):
    """
    A: (r_c, r_c), B: (r_c, num_chaos).
    あなたの関数と同等の見た目にします。save_path があればPNG保存。
    """
    colors = [(0, "blue"), (0.5, "white"), (1, "brown")]
    cmap = LinearSegmentedColormap.from_list("custom", colors)

    x_ticks = np.arange(-0.5, A.shape[1], 1)
    y_ticks = np.arange(-0.5, A.shape[0], 1)

    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], wspace=0.05)

    ax0 = plt.subplot(gs[0])
    im0 = ax0.imshow(A, cmap=cmap, vmin=A.min(), vmax=A.max())
    ax0.set_title("Matrix A")
    ax0.grid(True, color="black", linewidth=0.5)
    ax0.set_xticks(x_ticks)
    ax0.set_yticks(y_ticks)
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    ax0.tick_params(axis="both", which="both", length=0, labelsize=15)

    ax1 = plt.subplot(gs[1])
    ax1.imshow(B.reshape(-1, 1), cmap=cmap, vmin=A.min(), vmax=A.max())
    ax1.set_title("Vector B")
    ax1.grid(True, color="black", linewidth=0.5)
    ax1.set_xticks([])
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels([])
    ax1.tick_params(axis="both", which="both", length=0, labelsize=15)

    cbar = fig.colorbar(
        im0, ax=[ax0, ax1], orientation="vertical", fraction=0.02, pad=0.04
    )
    cbar.set_label("Value")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def get_active_slices(indices, min_jump_dist):
    """
    連続区間（インデックスの飛びが min_jump_dist より大きい位置で分割）を返す。
    出力: [(start_idx, end_idx), ...]
    """
    if len(indices) == 0:
        return []
    diffs = np.diff(indices)
    split_indices = np.where(diffs > min_jump_dist)[0]
    starts = [indices[0]] + [indices[i + 1] for i in split_indices]
    ends = [indices[i] for i in split_indices] + [indices[-1]]
    return list(zip(starts, ends))


def plot_timeseries_HAVOK_with_slices_save(
    out_dir,
    label,
    data_original,  # shape (r, T_seg)
    data_pred,  # shape (T_seg, r-1)  (lsim output)
    u,  # shape (num_chaos, T_seg)
    t,  # shape (T_seg,)
    r,  # truncated rank (modes count)
    threshold,  # symmetric threshold on |vr|
    min_jump_dist=1,
    segment_width=20.0,
):
    """
    あなたの関数と同じ見た目/ロジックで、各セグメントを PNG 保存します。
    保存名: timeseries_segment_{k}.png
    """
    N = min(
        len(t),
        int(data_pred.shape[0]),
        int(data_original.shape[1]),
        int(u.shape[1]),
    )
    t = np.asarray(t).reshape(-1)[:N]
    v1_original = np.asarray(data_original[0, :N])
    v1 = np.asarray(data_pred[:N, 0])
    vr = np.asarray(u[0, :N]).reshape(-1)
    vr2 = vr**2
    vr_min, vr_max = np.min(vr), np.max(vr)

    # 活性化インデックス → 連続スライス
    active_indices = np.where(np.abs(vr) > threshold)[0]
    if active_indices.size > 0:
        diffs = np.diff(active_indices)
        split_indices = np.where(diffs > min_jump_dist)[0]
        starts = [active_indices[0]] + [active_indices[i + 1] for i in split_indices]
        ends = [active_indices[i] for i in split_indices] + [active_indices[-1]]
        active_slices = list(zip(starts, ends))
    else:
        active_slices = []

    # ローブスイッチ（符号変化点）
    lobe_switch_indices = np.where(np.diff(np.sign(v1)) != 0)[0] + 1

    t_min, t_max = float(t[0]), float(t[-1])

    if segment_width is None or segment_width <= 0:
        segment_edges = np.array([t_min, t_max], dtype=float)
    else:
        n_seg = max(1, int(np.ceil((t_max - t_min) / segment_width)))
        segment_edges = np.linspace(t_min, t_max, n_seg + 1)

    vr_min, vr_max = float(np.min(vr)), float(np.max(vr))
    if vr_min == vr_max:
        vr_min, vr_max = (vr_min - 1.0,)

    for k in range(len(segment_edges) - 1):
        t_start = segment_edges[k]
        t_end = segment_edges[k + 1]
        mask = (t >= t_start) & (t < t_end)
        if not np.any(mask):
            continue

        t_seg = t[mask]
        v1_original_seg = v1_original[mask]
        v1_seg = v1[mask]
        vr_seg = vr[mask]
        vr2_seg = vr2[mask]

        # 区間内ローブスイッチ
        lobe_seg = lobe_switch_indices[
            (t[lobe_switch_indices] >= t_start) & (t[lobe_switch_indices] < t_end)
        ]

        # 区間内の活性スライス（ローカルインデックス）
        local_slices = []
        for s, e in active_slices:
            if t[s] > t_end or t[e] < t_start:
                continue
            start_idx = np.searchsorted(t_seg, t[s])
            end_idx = np.searchsorted(t_seg, t[e])
            if start_idx < len(t_seg) and end_idx <= len(t_seg):
                local_slices.append((start_idx, end_idx))

        # === プロット ===
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

        # v1（真値＆予測）
        axs[0].plot(
            t_seg, v1_original_seg, color="black", alpha=0.6, label="original v1"
        )
        axs[0].plot(
            t_seg,
            v1_seg,
            color="red",
            linestyle="--",
            alpha=0.6,
            label="HAVOK regression model",
        )
        axs[0].set_ylabel(r"$v_1$", fontsize=14)
        axs[0].legend(loc="upper right", fontsize=10)

        # v1（スイッチ＆活性表示）
        axs[1].plot(t_seg, v1_seg, color="gray", alpha=0.6)
        axs[1].plot(t[lobe_seg], v1[lobe_seg], "bx", label="Lobe Switch")
        for s, e in local_slices:
            axs[1].plot(t_seg[s:e], v1_seg[s:e], color="red", linewidth=2)
        axs[1].set_ylabel(r"$v_1$", fontsize=14)
        axs[1].legend(loc="upper right", fontsize=10)

        # vr
        axs[2].fill_between(
            t_seg, -threshold, threshold, color="gray", alpha=0.3, label="Threshold"
        )
        axs[2].plot(t_seg, vr_seg, color="black", alpha=0.3)
        for s, e in local_slices:
            axs[2].plot(t_seg[s:e], vr_seg[s:e], color="red", linewidth=1.5)
        axs[2].set_ylabel(rf"$v_{{{r}}}$", fontsize=14)
        axs[2].set_ylim(vr_min, vr_max)

        # vr^2
        axs[3].plot(t_seg, vr2_seg, color="black", alpha=0.3)
        for s, e in local_slices:
            axs[3].plot(t_seg[s:e], vr2_seg[s:e], color="red", linewidth=1.5)
        axs[3].set_ylabel(rf"$v_{{{r}}}^2$", fontsize=14)
        axs[3].set_xlabel("t", fontsize=14)
        axs[3].set_ylim(0.0, (vr_max**2) if vr_max > 0 else 1.0)

        for ax in axs:
            ax.tick_params(axis="both", labelsize=12)

        fig.suptitle(
            f"{label}: HAVOK Forcing (t={t_start:.3f}–{t_end:.3f})", fontsize=16
        )
        fig.tight_layout()

        save_path = os.path.join(out_dir, f"timeseries_segment_{k:02d}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def plot_attractor_vertical_with_activation_slices_save(
    out_dir, V_test, yout, u_test, t_test, threshold, min_jump_dist, label="Attractor"
):
    """
    あなたの3段縦連結のアトラクタ図を PNG 保存します。
    保存名: attractor.png
    """
    v1 = V_test[0, :]
    v2 = V_test[1, :]
    v3 = V_test[2, :]
    vr = u_test[0, :].flatten()
    active_indices = np.where(np.abs(vr) > threshold)[0]
    active_slices = get_active_slices(active_indices, min_jump_dist)

    v1_hat = yout[:, 0]
    v2_hat = yout[:, 1] if yout.shape[1] > 1 else np.zeros_like(v1_hat)
    v3_hat = yout[:, 2] if yout.shape[1] > 2 else np.zeros_like(v1_hat)

    fig = plt.figure(figsize=(7, 15))

    # 元アトラクタ
    ax1 = fig.add_subplot(3, 1, 1, projection="3d")
    ax1.plot(v1, v2, v3, color="black", alpha=0.8)
    ax1.set_title(f"{label} - Reconstruct Attractor", fontsize=14)
    ax1.set_xlabel("$v_1$")
    ax1.set_ylabel("$v_2$")
    ax1.set_zlabel("$v_3$")
    ax1.grid(False)

    # 活性区間を赤で重ね描き
    ax2 = fig.add_subplot(3, 1, 2, projection="3d")
    ax2.plot(v1, v2, v3, color="black", alpha=0.5, label="inactive")
    for s, e in active_slices:
        ax2.plot(v1[s:e], v2[s:e], v3[s:e], color="red", alpha=0.9)
    ax2.set_title(f"{label} - Forcing Activation Highlighted", fontsize=14)
    ax2.set_xlabel("$v_1$")
    ax2.set_ylabel("$v_2$")
    ax2.set_zlabel("$v_3$")
    ax2.grid(False)

    # HAVOK再構成アトラクタ
    ax3 = fig.add_subplot(3, 1, 3, projection="3d")
    ax3.plot(v1_hat, v2_hat, v3_hat, color="blue", alpha=0.3)
    ax3.set_title(f"{label} - HAVOK Model Reconstruction", fontsize=14)
    ax3.set_xlabel("$\\hat{v}_1$")
    ax3.set_ylabel("$\\hat{v}_2$")
    ax3.set_zlabel("$\\hat{v}_3$")
    ax3.grid(False)

    fig.tight_layout()
    save_path = os.path.join(out_dir, "attractor.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =========================
# Data loader (same as before)
# =========================


def generate_lorenz63(n_steps=200000, dt=0.01, x0=(1.0, 1.0, 1.0)):
    from scipy.integrate import solve_ivp

    def rhs(t, s, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
        x, y, z = s
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    t_span = (0, n_steps * dt)
    t_eval = np.linspace(*t_span, n_steps)
    sol = solve_ivp(
        rhs, t_span, x0, t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-12
    )
    return t_eval, sol.y.T[:, 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", type=str, default="")
    p.add_argument("--col", type=str, default="x")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--out_dir", type=str, default="results_hkAE_havok")

    # HKAE
    p.add_argument("--m", type=int, default=60)
    p.add_argument("--tau", type=int, default=1)
    p.add_argument("--latent_dim", type=int, default=8)  # Lorenz63では8
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)

    # HAVOK-on-latent
    p.add_argument("--m_havok", type=int, default=100)
    p.add_argument("--rank", type=int, default=8)  # Lorenz63では8
    p.add_argument("--havok_tau", type=int, default=1)
    p.add_argument("--num_chaos", type=int, default=1)

    # 可視化パラメータ（任意）
    p.add_argument("--segment_width", type=float, default=20.0)
    p.add_argument("--min_jump_sec", type=float, default=0.5)
    p.add_argument(
        "--time_model",
        type=str,
        default="continuous",
        choices=["continuous", "discrete"],
    )
    # （連続時間用のオプション。離散では無視されます）
    p.add_argument(
        "--diff_method", type=str, default="central", choices=["central", "savgol"]
    )
    p.add_argument("--savgol_window", type=int, default=11)
    p.add_argument("--savgol_poly", type=int, default=3)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--stabilize", action="store_true")
    p.add_argument("--no-stabilize", dest="stabilize", action="store_false")
    p.set_defaults(stabilize=True)
    p.add_argument("--stabilize_eps", type=float, default=1e-6)

    args, _ = p.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 連続/離散でサブフォルダを分ける（例：results_hkAE_havok/continuous や /discrete）
    out_dir = os.path.join(args.out_dir, args.time_model)
    os.makedirs(out_dir, exist_ok=True)

    # data
    # data
    if args.data_csv and os.path.exists(args.data_csv):
        if bh_load_csv_data is None:
            raise RuntimeError(
                "build_havok.load_csv_data が見つかりません。havok/build_havok.py を確認してください。"
            )
        # 単一CSVを基本とし、カンマ区切りなら最初の1つを使用（必要なら拡張）
        first_csv = args.data_csv.split(",")[0].strip()
        arr = bh_load_csv_data([first_csv])[0]  # ndarray

        # 期待形状に整形：1D→そのまま，2D→列選択（col が整数ならその列，そうでなければ 0 列）
        if arr.ndim == 1:
            x = arr.astype(float)
        else:
            # --col が "0","1" のような数字ならその列を使う。それ以外は 0 列を使用
            try:
                col_idx = int(args.col)
            except Exception:
                col_idx = 0
            if col_idx < 0 or col_idx >= arr.shape[1]:
                raise ValueError(
                    f"--col の列番号が範囲外です: {col_idx} (列数={arr.shape[1]})"
                )
            x = arr[:, col_idx].astype(float)

        t = np.arange(x.shape[0]) * args.dt
    else:
        t, x = generate_lorenz63(n_steps=200000, dt=args.dt)

    # normalize
    x = (x - x.mean()) / (x.std() + 1e-12)

    # HKAE Hankel
    xZ = np.vstack([x]).T  # (T,1)
    H = delay_stack(xZ, m=args.m, tau=args.tau)  # (m, L)

    split = int(H.shape[1] * 0.7)
    H_train, H_val = H[:, :split], H[:, split:]

    hcfg = HKAEConfig(
        m=args.m,
        tau=args.tau,
        latent_dim=args.latent_dim,
        hidden=args.hidden,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    trainer = HKAETrainer(hcfg)
    model = trainer.fit(H_train, H_val)

    # # HAVOK on latent (continuous-time)
    # vcfg = HavokConfig(
    #     m_havok=args.m_havok,
    #     tau=args.havok_tau,
    #     rank=args.rank,
    #     horizon=args.horizon,
    #     num_chaos=args.num_chaos,
    #     dt=args.dt,
    #     time_model=args.time_model,
    #     ridge=args.ridge,
    # )
    # vcfg.diff_method = getattr(args, "diff_method", "savgol")
    # vcfg.savgol_window = getattr(args, "savgol_window", 11)
    # vcfg.savgol_poly = getattr(args, "savgol_poly", 3)
    # vcfg.stabilize = getattr(args, "stabilize", True)
    # vcfg.stabilize_eps = getattr(args, "stabilize_eps", 1e-6)
    # vcfg.train_frac = getattr(args, "train_frac", 0.8)

    # HAVOK on latent (discreate-time)
    vcfg = HavokConfig(
        m_havok=args.m_havok,
        tau=args.havok_tau,
        rank=args.rank,
        horizon=args.horizon,
        num_chaos=args.num_chaos,
        dt=args.dt,
        time_model="discrete",
        ridge=args.ridge,
    )

    analyzer = HavokLatentAnalyzer(vcfg)
    res = analyzer.analyze(model, H, device=hcfg.device)
    # --- align for plotting (add this) ---
    # 解析結果から境界を取得（なければ 0.8L）
    L = res["V"].shape[1]
    idx_train = int(res.get("idx_train", int(0.8 * L)))

    t_test = res["t_test"]
    yout = res["yout"]  # (T_pred, r_c)
    u_test = res["u_test"]  # (num_chaos, T_pred?)

    # 長さを最小に合わせる
    Lmin = min(yout.shape[0], u_test.shape[1], t_test.shape[0])
    yout = yout[:Lmin, :]
    u_test = u_test[:, :Lmin]
    t_test = t_test[:Lmin]

    # ★ V_test は「更新後の状態」に合わせて +1 から開始
    V_full = res["V"]
    V_test = V_full[:, idx_train + 1 : idx_train + 1 + Lmin]

    # デバッグ出力（任意）
    print(
        "aligned:",
        "t_test",
        t_test.shape,
        "yout",
        yout.shape,
        "u_test",
        u_test.shape,
        "V_test",
        V_test.shape,
    )
    # --- end align ---

    # 係数の扱いを time_model で分岐
    if args.time_model == "discrete":
        A_d, B_d = res["A"], res["B"]
        # 離散→連続（可視化・検証用）
        A_c, B_c = discrete_to_continuous(A_d, B_d, args.dt)
    else:
        # 連続推定の結果
        A_c, B_c = res["A"], res["B"]
        # 連続→離散（検証・一部関数の互換用）
        n, m = A_c.shape[0], B_c.shape[1]
        Phi_c = np.block([[A_c, B_c], [np.zeros((m, n)), np.zeros((m, m))]])
        Phi_d = expm(args.dt * Phi_c)
        A_d, B_d = Phi_d[:n, :n], Phi_d[:n, n:]

    # 保存
    np.save(os.path.join(out_dir, "A_cont.npy"), A_c)
    np.save(os.path.join(out_dir, "B_cont.npy"), B_c)

    # 可視化（ゼロ中心の対称カラースケール推奨）
    def _plot_AB_zero_center(A, B, save_path):
        vlim = float(np.max(np.abs(A)))
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], wspace=0.05)
        ax0 = plt.subplot(gs[0])
        im0 = ax0.imshow(A, cmap="bwr", vmin=-vlim, vmax=vlim)
        ax0.set_title("Matrix A (continuous)")
        ax0.grid(True, color="black", linewidth=0.5)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1 = plt.subplot(gs[1])
        ax1.imshow(B.reshape(-1, 1), cmap="bwr", vmin=-vlim, vmax=vlim)
        ax1.set_title("Vector B (continuous)")
        ax1.grid(True, color="black", linewidth=0.5)
        ax1.set_xticks([])
        ax1.set_yticks([])
        cbar = fig.colorbar(
            im0, ax=[ax0, ax1], orientation="vertical", fraction=0.02, pad=0.04
        )
        cbar.set_label("Value")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    # 既存の plot_AB があればそれ＋連続版の PNG も保存
    if plot_AB is not None:
        try:
            plot_AB(A_c, B_c)
        except Exception:
            pass
    _plot_AB_zero_center(A_c, B_c, os.path.join(out_dir, "AB_params_cont.png"))

    # --- 連続→離散の往復誤差で検証（合っているほど小さい） ---
    n, m = A_c.shape[0], B_c.shape[1]
    if args.time_model == "discrete":
        # d -> c -> d
        Phi_c = np.block([[A_c, B_c], [np.zeros((m, n)), np.zeros((m, m))]])
        Phi_d_hat = expm(args.dt * Phi_c)
        A_d_hat, B_d_hat = Phi_d_hat[:n, :n], Phi_d_hat[:n, n:]
        errA = np.linalg.norm(A_d - A_d_hat, "fro") / (
            np.linalg.norm(A_d, "fro") + 1e-12
        )
        errB = np.linalg.norm(B_d - B_d_hat, "fro") / (
            np.linalg.norm(B_d, "fro") + 1e-12
        )
    else:
        # c -> d -> c
        Phi_d = np.block([[A_d, B_d], [np.zeros((m, n)), np.eye(m)]])
        from havok.post_latent import discrete_to_continuous

        A_c_hat, B_c_hat = discrete_to_continuous(A_d, B_d, args.dt)
        errA = np.linalg.norm(A_c - A_c_hat, "fro") / (
            np.linalg.norm(A_c, "fro") + 1e-12
        )
        errB = np.linalg.norm(B_c - B_c_hat, "fro") / (
            np.linalg.norm(B_c, "fro") + 1e-12
        )
    print(
        f"[check] discrete->continuous->discrete errors:  A: {errA:.2e},  B: {errB:.2e}"
    )

    # -----------------------------
    # 保存: A, B, V, forcing
    # -----------------------------
    np.save(os.path.join(out_dir, "A.npy"), res["A"])
    np.save(os.path.join(out_dir, "B.npy"), res["B"])
    np.save(os.path.join(out_dir, "V.npy"), res["V"])
    np.save(os.path.join(out_dir, "forcing.npy"), res["V"][-args.num_chaos :, :])

    cfg_dict = vars(args)
    np.save(os.path.join(out_dir, "config.npy"), cfg_dict, allow_pickle=True)
    # -----------------------------
    # ① A/B の形状とプロット
    # -----------------------------
    A = res["A"]
    B = res["B"]
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")

    # ユーザー関数があれば表示、さらにPNGも保存
    if plot_AB is not None:
        try:
            plot_AB(A, B)
        except Exception:
            pass
    plot_matrix_and_vector_with_shared_colorbar_fallback(
        A, B, save_path=os.path.join(out_dir, "AB_params.png")
    )

    # -----------------------------
    # ② 〜 ③ あなたの“活性スライス赤表示”に準拠した図
    # -----------------------------
    # analyzer では train: 0〜0.8L, test: 0.8L〜 で lsim を実施
    V_full = res["V"]
    L = V_full.shape[1]
    idx_train = int(res.get("idx_train", int(0.8 * L)))

    # t_test, yout, u_test は analyzer が返している
    t_test = res["t_test"]
    yout = res["yout"]  # shape (T_test, r-1)
    u_test = res["u_test"]  # shape (num_chaos, T_test)
    r = res["r"]  # truncated rank

    # V_test を yout/u_test と長さを合わせて切り出し
    # T_test = yout.shape[0]
    # V_test = V_full[:, idx_train + 1 : idx_train + T_test]

    T_test = yout.shape[0]
    if args.time_model == "discrete":
        V_test = V_full[:, idx_train + 1 : idx_train + 1 + T_test]
    else:
        V_test = V_full[:, idx_train : idx_train + T_test]

    # 閾値（±95%タイル）
    thr = compute_forcing_threshold(u_test, percentile=95)

    # スライスの幅（時間単位）とジャンプ距離（サンプル数）
    segment_width = args.segment_width
    min_jump_dist = max(1, int(args.min_jump_sec / args.dt))

    # 時系列（v1/vr/vr^2）: 区間ごとに PNG 保存
    plot_timeseries_HAVOK_with_slices_save(
        out_dir=out_dir,
        label="Latent-HAVOK",
        data_original=V_test,
        data_pred=yout,
        u=u_test,
        t=t_test,
        r=r,
        threshold=thr,
        min_jump_dist=min_jump_dist,
        segment_width=segment_width,
    )

    # 3D アトラクタ（元/活性強調/再構成）
    if V_test.shape[0] >= 3:
        plot_attractor_vertical_with_activation_slices_save(
            out_dir=out_dir,
            V_test=V_test,
            yout=yout,
            u_test=u_test,
            t_test=t_test,
            threshold=thr,
            min_jump_dist=min_jump_dist,
            label="Latent-HAVOK",
        )

    print("[done] Saved to", out_dir)

    # 予測の復元：潜在HAVOK → 観測空間
    bp_cfg = BackProjectConfig(
        m_orig=args.m,
        m_havok=args.m_havok,
        dt=args.dt,
    )
    t_pred, x_pred, stats = predict_observation_from_latent(
        model=model, res=res, U=res["U"], s=res["s"], cfg=bp_cfg
    )
    np.save(os.path.join(out_dir, "t_pred.npy"), t_pred)
    np.save(os.path.join(out_dir, "x_pred_from_latent_havok.npy"), x_pred)

    # 参考用にプロットも保存
    fig = plt.figure(figsize=(10, 3))
    plt.plot(t[: len(x)], x, label="x (normalized)")
    plt.plot(t_pred, x_pred, "--", label="x_pred (latent-HAVOK)")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("x")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "x_pred_overlay.png"), dpi=150)
    plt.close(fig)

    # 妥当性指標（JSONで保存＆標準出力で要約）
    A_d, B_d = res["A"], res["B"]
    metrics = metrics_latent_havok(
        res=res,
        A_d=A_d,
        B_d=B_d,
        dt=args.dt,
        V_full=res["V"],
        x_ref=x[: len(t)],
        t_ref=t,
        x_pred=x_pred,
        t_pred=t_pred,
        percentile=95.0,
    )

    np.save(
        os.path.join(out_dir, "metrics_latent_havok.npy"), metrics, allow_pickle=True
    )

    out = compare_timeseries_and_plot(
        t_ref=t,
        x_ref=x,
        t_pred=t_pred,
        x_pred=x_pred,
        dt=None,  # 省略可：自動で中央値差
        out_dir=os.path.join(out_dir, "compare"),  # 保存したい場合
        filename_base="obs",
        active_intervals=None,  # [(ts, te), ...] を渡すと赤帯表示
        title="Original vs Predicted",
    )

    metrics = show_learning_metrics(
        out["x_ref_grid"],
        out["x_pred_grid"],
        dt=np.median(np.diff(out["t_grid"])),
        out_dir=os.path.join(out_dir, "compare"),
        filename="metrics_observation",
    )

    # 追加：5分割で図＆npy保存
    _ = compare_timeseries_and_plot_segmented(
        t_ref=t,
        x_ref=x,
        t_pred=t_pred,
        x_pred=x_pred,
        n_segments=5,  # ★ ここを変えれば分割数を調整
        dt=None,
        out_dir=os.path.join(out_dir, "compare"),
        filename_base="obs",  # 出力は obs_seg00.png, obs_seg01.png, ...
        active_intervals=None,  # あれば [(ts, te), ...]
        title_prefix="Original vs Predicted",
    )


if __name__ == "__main__":
    main()
