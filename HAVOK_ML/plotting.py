import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap


def plot_matrix_and_vector_with_shared_colorbar(A, B):
    """
    HAVOKモデルのパラメータの可視化
    赤と青で正負を表現
    白色はゼロを表現
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

    plt.show()


def get_active_slices(indices, min_jump_dist):
    if len(indices) == 0:
        return []
    diffs = np.diff(indices)
    splits = np.where(diffs > min_jump_dist)[0]
    starts = [indices[0]] + [indices[i + 1] for i in splits]
    ends = [indices[i] for i in splits] + [indices[-1]]
    return list(zip(starts, ends))


def plot_timeseries_HAVOK_with_slices(
    label,
    data_original,
    data,
    u,  # 強制項（固有時間遅れ座標のr番目の要素vrの時間発展）
    t,
    r,
    threshold=0.002,  # 強制項閾値
    min_jump_dist=1,
    true_switch_indices=None,
    segment_width=50,
):
    """
    HAVOK時系列データに対して、活性化（Forcing Active）区間、ローブスイッチなどを視覚化。
    `get_active_slices` を用いた活性区間抽出に対応。

    Parameters:
    - label: グラフタイトル
    - data: HAVOK出力の時系列データ（形状: [N, r]）
    - t: 時間軸
    - r: 強制項のモードインデックス（1始まり）
    - threshold: 活性と判定する閾値（|vᵣ| > threshold）
    - min_jump_dist: 活性のジャンプ判定に用いる最小インデックス差
    - true_switch_indices: 実ローブスイッチのインデックス（オプション）
    """

    v1_original = data_original[0, :]
    v1 = data[:, 0]
    vr = u[0].flatten()
    vr2 = vr**2
    vr_min, vr_max = np.min(vr), np.max(vr)

    # 活性インデックスとその連続区間を取得
    active_indices = np.where(np.abs(vr) > threshold)[0]
    active_slices = get_active_slices(active_indices, min_jump_dist)

    # ローブスイッチの検出
    lobe_switch_indices = np.where(np.diff(np.sign(v1)) != 0)[0] + 1

    t_min, t_max = t[0], t[-1]
    segment_edges = np.arange(t_min, t_max + segment_width, segment_width)

    for i in range(len(segment_edges) - 1):
        t_start = segment_edges[i]
        t_end = segment_edges[i + 1]

        # 区間に対応するインデックス抽出
        mask = (t >= t_start) & (t < t_end)
        t_seg = t[mask]
        v1_original_seg = v1_original[mask]
        v1_seg = v1[mask]
        vr_seg = vr[mask]
        vr2_seg = vr2[mask]

        # 区間内のローブスイッチ
        lobe_seg = lobe_switch_indices[
            (t[lobe_switch_indices] >= t_start) & (t[lobe_switch_indices] < t_end)
        ]

        # 区間内の活性スライス（インデックス変換済み）
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

        # === v1 ===
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

        # === v1 ===
        axs[1].plot(t_seg, v1_seg, color="gray", alpha=0.6)
        axs[1].plot(t[lobe_seg], v1[lobe_seg], "bx", label="Lobe Switch")
        for s, e in local_slices:
            axs[1].plot(t_seg[s:e], v1_seg[s:e], color="red", linewidth=2)
        axs[1].set_ylabel(r"$v_1$", fontsize=14)
        axs[1].legend(loc="upper right", fontsize=10)

        # === vr ===
        axs[2].fill_between(
            t_seg, -threshold, threshold, color="gray", alpha=0.3, label="Threshold"
        )
        axs[2].plot(t_seg, vr_seg, color="black", alpha=0.3)
        for s, e in local_slices:
            axs[2].plot(t_seg[s:e], vr_seg[s:e], color="red", linewidth=1.5)
        axs[2].set_ylabel(rf"$v_{{{r}}}$", fontsize=14)
        axs[2].set_ylim(vr_min, vr_max)  # 縦軸スケーリングを固定

        # === vr^2 ===
        axs[3].plot(t_seg, vr2_seg, color="black", alpha=0.3)
        for s, e in local_slices:
            axs[3].plot(t_seg[s:e], vr2_seg[s:e], color="red", linewidth=1.5)
        axs[3].set_ylabel(rf"$v_{{{r}}}^2$", fontsize=14)
        axs[3].set_xlabel("t", fontsize=14)
        axs[3].set_ylim(0.0, vr_max**2)  # 縦軸スケーリングを固定

        for ax in axs:
            ax.tick_params(axis="both", labelsize=12)

        fig.suptitle(
            f"{label}: HAVOK Forcing (t={t_start:.0f}–{t_end:.0f})", fontsize=16
        )
        fig.tight_layout()
        plt.show()


def plot_attractor_vertical_with_activation_slices(
    V_test, yout, u_test, t_test, threshold, min_jump_dist, label="Attractor"
):
    """
    HAVOKアトラクタを縦に3分割で表示：
    - 元の固有遅れ座標によるアトラクタ
    - 活性/非活性による色分け付きアトラクタ（スライス処理による）
    - HAVOKモデル出力によるアトラクタ

    Parameters:
    - V_test: [r, T] SVDから得た遅れ座標行列
    - yout: [T, r-1] HAVOKモデルからの解析解
    - u_test: [1, T] 強制項（v_r）
    - t_test: 時間軸（1次元配列）
    - threshold: 活性のしきい値（|v_r| > threshold）
    - min_jump_dist: 活性とみなす区間の最小ギャップ（index単位）
    - label: タイトル接頭辞
    """
    v1 = V_test[0, :]
    v2 = V_test[1, :]
    v3 = V_test[2, :]
    vr = u_test[0, :].flatten()
    active_indices = np.where(np.abs(vr) > threshold)[0]
    active_slices = get_active_slices(active_indices, min_jump_dist)

    v1_hat = yout[:, 0]
    v2_hat = yout[:, 1]
    v3_hat = yout[:, 2]

    fig = plt.figure(figsize=(7, 15))

    # === 元のアトラクタ ===
    ax1 = fig.add_subplot(3, 1, 1, projection="3d")
    ax1.plot(v1, v2, v3, color="black", alpha=0.8)
    ax1.set_title(f"{label} - Reconstruct Attractor", fontsize=14)
    ax1.set_xlabel("$v_1$")
    ax1.set_ylabel("$v_2$")
    ax1.set_zlabel("$v_3$")
    ax1.grid(False)

    # === 活性/非活性を区別したアトラクタ（スライス処理で連続区間描画）===
    ax2 = fig.add_subplot(3, 1, 2, projection="3d")
    ax2.plot(v1, v2, v3, color="black", alpha=0.5, label="inactive")
    for s, e in active_slices:
        ax2.plot(v1[s:e], v2[s:e], v3[s:e], color="red", alpha=0.9)
    ax2.set_title(f"{label} - Forcing Activation Highlighted", fontsize=14)
    ax2.set_xlabel("$v_1$")
    ax2.set_ylabel("$v_2$")
    ax2.set_zlabel("$v_3$")
    ax2.grid(False)

    # === HAVOKモデルによる再構成アトラクタ ===
    ax3 = fig.add_subplot(3, 1, 3, projection="3d")
    ax3.plot(v1_hat, v2_hat, v3_hat, color="blue", alpha=0.3)
    ax3.set_title(f"{label} - HAVOK Model Reconstruction", fontsize=14)
    ax3.set_xlabel("$\\hat{v}_1$")
    ax3.set_ylabel("$\\hat{v}_2$")
    ax3.set_zlabel("$\\hat{v}_3$")
    ax3.grid(False)

    plt.tight_layout()
    plt.show()


def plot_havok_reconstruction(t_rec, V, yout_pred, u_pred, r, idx_train_end, fig=None):
    """
    Multi-step HAVOK 再構成結果を 2 行プロット (v1 と v_r)

    Parameters
    ----------
    t_rec : array-like, shape (N,)
        再構成期間の時刻ベクトル
    V : array-like, shape (r, M)
        HAVOK モード時系列 (真の値), M >= idx_train_end + N
    yout_pred : array-like, shape (N, r-1)
        予測モード時系列 (lsim からの出力)
    u_pred : array-like, shape (N,)
        予測 forcing term (vr)
    r : int
        モード数
    idx_train_end : int
        訓練終了インデックス (再構成開始位置)
    fig : matplotlib.figure.Figure, optional
        既存の Figure を利用する場合

    Returns
    -------
    fig, axs : tuple
        Figure オブジェクトと Axes の配列 (2 要素)
    """
    if fig is None:
        fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    else:
        axs = fig.get_axes()
    N = len(t_rec)

    # v1 (mode 1)
    true_v1 = V[0, idx_train_end : idx_train_end + N]
    recon_v1 = yout_pred[:, 0]
    axs[0].plot(t_rec, true_v1, label="true v1")
    axs[0].plot(t_rec, recon_v1, "--", label="recon v1")
    axs[0].set_ylabel(r"$v_1$", fontsize=14)
    axs[0].legend()

    # v_r (forcing mode)
    true_vr = V[r - 1, idx_train_end : idx_train_end + N]
    recon_vr = u_pred
    axs[1].plot(t_rec, true_vr, label=f"true v{{{r}}}")
    axs[1].plot(t_rec, recon_vr, "--", label="recon vr")
    axs[1].set_ylabel(rf"$v_{{{r}}}$", fontsize=14)
    axs[1].legend()

    fig.suptitle("Multi-step HAVOK reconstruction", fontsize=16)
    fig.tight_layout()
    return fig, axs
