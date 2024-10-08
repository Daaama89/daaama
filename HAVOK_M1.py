import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import StateSpace, lsim


# Load CSV data
def load_csv_data(filenames, delimiter=",", dtype=float, encoding="ms932"):
    return [
        np.genfromtxt(file, delimiter=delimiter, dtype=dtype, encoding=encoding)
        for file in filenames
    ]


# Create a Hankel matrix
def create_hankel_matrix(data, m):
    if len(data) == 0:
        raise ValueError("Data array is empty.")

    k = len(data)
    if m > k:
        raise ValueError(f"m (={m}) is larger than the length of the data (={k}).")

    n = k - m + 1
    hankeldata = np.column_stack([data[i : i + m] for i in range(0, n)])
    hankel_matrix = np.array(hankeldata)
    return hankel_matrix


# Perform SVD and return truncated matrices
def truncated_svd(matrix, rank):
    U, S, Vt = sp.linalg.svd(matrix, full_matrices=0)
    return U[:, :rank], np.diag(S[:rank]), Vt[:rank, :]


# Create time-shifted matrix
def create_difference_matrix(n, X):
    identity_matrix = np.identity(n)
    difference_matrix = X - identity_matrix[:n, :n]
    return difference_matrix


# Create linear state-space model
def create_linear_model(A, r):
    A_dp = A[: r - 1, : r - 1]
    B_dp = A[:-1, r - 1].reshape(-1, 1)
    plot_matrix_and_vector_with_shared_colorbar(A_dp, B_dp)
    sys_new = StateSpace(A_dp, B_dp, np.eye(r - 1), np.zeros((r - 1, 1)))
    return sys_new


# Simulate the linear system
def simulate_system(sys, input_data, t, initial_conditions):
    t, y, _ = lsim(sys, input_data, t, initial_conditions)
    return t, y


# Plot the Matrix and bector
def plot_matrix_and_vector_with_shared_colorbar(A, B):
    # colormap
    colors = [(0, "blue"), (0.5, "white"), (1, "brown")]
    cmap = LinearSegmentedColormap.from_list("custom", colors)

    # ticks settings
    x_ticks = np.arange(-0.5, A.shape[1], 1)
    y_ticks = np.arange(-0.5, A.shape[0], 1)

    # make plot
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], wspace=0.05)

    # matrix plot
    ax0 = plt.subplot(gs[0])
    im0 = ax0.imshow(A, cmap=cmap, vmin=A.min(), vmax=A.max())
    ax0.set_title("Matrix A")
    ax0.grid(True, color="black", linewidth=0.5)
    ax0.set_xticks(x_ticks)
    ax0.set_yticks(y_ticks)
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    ax0.tick_params(axis="both", which="both", length=0, labelsize=15)

    # bector plot
    ax1 = plt.subplot(gs[1])
    ax1.imshow(B.reshape(-1, 1), cmap=cmap, vmin=A.min(), vmax=A.max())
    ax1.set_title("Vector B")
    ax1.grid(True, color="black", linewidth=0.5)
    ax1.set_xticks([])
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels([])
    ax1.tick_params(axis="both", which="both", length=0, labelsize=15)

    # add the color bar
    cbar = fig.colorbar(
        im0, ax=[ax0, ax1], orientation="vertical", fraction=0.02, pad=0.04
    )
    cbar.set_label("Value")

    plt.show()


# run HAVOK lorenz and so onn
def run_havok_analysis(data, m, r, dt, label):
    # Create Hankel matrix
    hankel_matrix = create_hankel_matrix(data, m)

    # Perform SVD
    U, S, Vt = truncated_svd(hankel_matrix, r)

    # Time-shifted matrices
    V_1 = Vt[:, :-1].T
    V_2 = Vt[:, 1:].T

    # Linear approximation A-hat
    A_1 = np.linalg.lstsq(V_2, V_1, rcond=None)[0]
    result_matrix = create_difference_matrix(A_1.shape[0], A_1) / dt

    # Create state-space model
    sys_new = create_linear_model(result_matrix, r)

    # Simulate the system
    V = Vt.T
    t_span, y = simulate_system(
        sys_new,
        V[:, r - 1],
        np.arange(0, len(V) * dt, dt),
        V[0, : r - 1],
    )

    # Plot results
    plot_embedding(V, label, dt)
    plot_attractor_forcing_active(V, r, dt)
    plot_time_series_forcing_active(V, dt, r)


def plot_results(V, y, dt, r, label):
    n_plots = y.shape[1]
    plots_per_fig = 4
    t = np.arange(0, len(V) * dt, dt)

    for start_idx in range(0, n_plots, plots_per_fig):
        end_idx = min(start_idx + plots_per_fig, n_plots)
        fig, axs = plt.subplots(end_idx - start_idx, 1, sharex=True, figsize=(7, 9))

        for i in range(start_idx, end_idx):
            ax = axs[i - start_idx]
            ax.plot(
                t[500000:550000],
                V[500000:550000, i],
                "k",
                label=f"{label} eigen-time-delay-coordinate",
            )
            ax.plot(
                t[500000:550000],
                y[500000:550000, i],
                "r--",
                label=f"{label} HAVOK regression model",
            )
            ax.legend()
            ax.set(xlabel="t", ylabel=r"$v_{}$".format(i + 1))

        plt.tight_layout()
        plt.show()


def plot_embedding(V, label, dt):

    t_length = np.arange(0, len(V) // 5 * dt, dt)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(V[: len(t_length), 0], V[: len(t_length), 1], V[: len(t_length), 2], "k")
    ax.set_xlabel("v1")
    ax.set_ylabel("v2")
    ax.set_zlabel("v3")
    ax.view_init(elev=35, azim=45)
    plt.tight_layout()
    plt.show()


def plot_attractor_forcing_active(V, r, dt):
    L = np.arange(len(V))
    inds = V[L, r - 1] ** 2 > 1.0e-6
    L = L[inds]

    startvals = []
    endvals = []
    start = 0
    numhits = 100

    for k in range(numhits):
        startvals.append(start)
        endmax = start + 500
        interval = np.arange(start, endmax)
        hits = np.where(inds[interval])[0]

        if hits.size > 0:
            endval = start + hits[-1]
            endvals.append(endval)
            newhit = np.where(inds[endval + 1 :])[0]

            if newhit.size > 0:
                start = endval + newhit[0]
            else:
                print("新しいヒットが見つかりませんでした。ループを終了します。")
        else:
            print("ヒットが見つかりませんでした。ループを終了します。")

    print("startvals, endvals = :", len(startvals), len(endvals))

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # プロット前にサイズを確認
    assert len(startvals) == len(
        endvals
    ), "startvals と endvals のサイズが一致しません。"

    for k in range(numhits):
        if k < len(startvals) and k < len(endvals):
            print(f"k = {k+1}, startval = {startvals[k]}, endval = {endvals[k]}")
            if startvals[k] < len(V) and endvals[k] < len(V):
                ax.plot(
                    V[startvals[k] : endvals[k], 0],
                    V[startvals[k] : endvals[k], 1],
                    V[startvals[k] : endvals[k], 2],
                    "r",
                    linewidth=1.5,
                )
            else:
                print("エラー: インデックスが範囲外です。")
        else:
            print("エラー: 配列のサイズが不足しています。")

    for k in range(numhits - 1):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax.plot(
                    V[endvals[k] : startvals[k + 1], 0],
                    V[endvals[k] : startvals[k + 1], 1],
                    V[endvals[k] : startvals[k + 1], 2],
                    color=[0.25, 0.25, 0.25],
                    linewidth=1.5,
                )
            else:
                print("エラー: インデックスが範囲外です。")
        else:
            print("エラー: 配列のサイズが不足しています。")

    # プロットのその他の設定
    ax.set_xlabel("v_1")
    ax.set_ylabel("v_2")
    ax.set_zlabel("v_3")

    plt.tight_layout()
    fig.set_size_inches(10, 10)
    ax.view_init(elev=0, azim=150)
    plt.show()


def plot_time_series_forcing_active(V, dt, r):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 9))
    tspan = np.arange(0, len(V) * dt, dt)
    L = np.arange(len(V))
    inds = V[L, r - 1] ** 2 > 1.0e-6
    L = L[inds]

    startvals = []
    endvals = []
    start = 0
    numhits = 100

    for k in range(numhits):
        startvals.append(start)
        endmax = start + 500
        interval = np.arange(start, endmax)
        hits = np.where(inds[interval])[0]

        if hits.size > 0:
            endval = start + hits[-1]
            endvals.append(endval)
            newhit = np.where(inds[endval + 1 :])[0]

            if newhit.size > 0:
                start = endval + newhit[0]
            else:
                print("新しいヒットが見つかりませんでした。ループを終了します。")
        else:
            print("ヒットが見つかりませんでした。ループを終了します。")

    # サブプロット1: v_1 の時系列データ
    ax1.plot(tspan[: len(V)], V[:, 0], "k")
    for k in range(numhits):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax1.plot(
                    tspan[startvals[k] : endvals[k]],
                    V[startvals[k] : endvals[k], 0],
                    "r",
                    linewidth=1.5,
                )
            else:
                print("エラー：インデックスが範囲外")
        else:
            print("エラー：配列のサイズが不足")

    for k in range(numhits - 1):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax1.plot(
                    tspan[endvals[k] : startvals[k + 1]],
                    V[endvals[k] : startvals[k + 1], 0],
                    color=[0.25, 0.25, 0.25],
                    linewidth=1.5,
                )
            else:
                print("エラー：インデックスが範囲外")
        else:
            print("エラー：配列のサイズが不足")

    ax1.set_xlim([-35, 153])
    ax1.set_ylim([-0.0047, 0.0027])
    ax1.set_ylabel("v_1")

    # サブプロット2: v_{r} の時系列データ
    ax2.plot(tspan[: len(V)], V[:, r - 1], "k")
    for k in range(numhits):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax2.plot(
                    tspan[startvals[k] : endvals[k]],
                    V[startvals[k] : endvals[k], r - 1],
                    "r",
                    linewidth=1.5,
                )
            else:
                print("エラー：インデックスが範囲外")
        else:
            print("エラー：配列のサイズが不足")

    for k in range(numhits - 1):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax2.plot(
                    tspan[endvals[k] : startvals[k + 1]],
                    V[endvals[k] : startvals[k + 1], r - 1],
                    color=[0.25, 0.25, 0.25],
                    linewidth=1.5,
                )
            else:
                print("エラー：インデックスが範囲外")
        else:
            print("エラー：配列のサイズが不足")

    ax2.set_ylabel(f"v_{r}")

    # サブプロット3: v_{15}^2 の時系列データ
    ax3.plot(tspan[startvals[0]], V[startvals[0], r - 1], "r")
    ax3.plot(tspan[endvals[0]], V[startvals[1], r - 1], color=[0.25, 0.25, 0.25])
    for k in range(numhits):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax3.plot(
                    tspan[startvals[k] : endvals[k]],
                    V[startvals[k] : endvals[k], r - 1] ** 2,
                    "r",
                    linewidth=1.5,
                )
            else:
                print("エラー：インデックスが範囲外")
        else:
            print("エラー：配列のサイズが不足")

    for k in range(numhits - 1):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax3.plot(
                    tspan[endvals[k] : startvals[k + 1]],
                    V[endvals[k] : startvals[k + 1], r - 1] ** 2,
                    color=[0.25, 0.25, 0.25],
                    linewidth=1.5,
                )
            else:
                print("エラー：インデックスが範囲外")
        else:
            print("エラー：配列のサイズが不足")

    ax3.set_xlabel("t")
    ax3.set_ylabel(f"v_{r}^2")
    ax3.set_ylim([0, 0.00025])

    # 共通の設定
    ax3.legend(["Forcing Active", "Forcing Inactive"])
    ax1.get_shared_x_axes().join(ax1, ax2, ax3)
    ax1.set_xlim([25, 65])

    # グラフのサイズや出力設定
    fig.tight_layout()
    plt.show()


def run_havok_analysis_double_pendulum(data, m, r, dt, label):
    # Create Hankel matrix
    hankel_matrix = create_hankel_matrix(data, m)

    # Perform SVD
    U, S, Vt = truncated_svd(hankel_matrix, r)

    # Time-shifted matrices
    V_1 = Vt[:, :-1].T
    V_2 = Vt[:, 1:].T

    # Linear approximation A-hat
    A_1 = np.linalg.lstsq(V_2, V_1, rcond=None)[0]
    result_matrix = create_difference_matrix(A_1.shape[0], A_1) / dt

    # Create state-space model
    sys_new = create_linear_model(result_matrix, r)

    # Simulate the system
    V = Vt.T
    t_span, y = simulate_system(
        sys_new,
        V[:, r - 1],
        np.arange(0, len(V) * dt, dt),
        V[0, : r - 1],
    )

    # Plot results
    plot_embedding(V, label, dt)
    plot_attractor_forcing_active_double_pendulum(V, r, dt)
    plot_time_series_forcing_active_double_pendulum(V, dt, r)

    np.savetxt("dp_V1.csv", V[:, 0], delimiter=",", fmt="%.18f")
    np.savetxt("dp_V2.csv", V[:, 1], delimiter=",", fmt="%.18f")
    np.savetxt("dp_V3.csv", V[:, 2], delimiter=",", fmt="%.18f")


def plot_attractor_forcing_active_double_pendulum(V, r, dt):
    # 信号
    L = np.arange(0, len(V) // 5)
    signal = V[L, r - 1] ** 2

    # 閾値を使った検出 (find_peaks の代わりに直接判定)
    threshold = 1.0e-5
    above_threshold = signal > threshold  # 閾値を超える部分はTrue

    # 閾値以下の部分も補完するためのロジック
    # 信号全体で閾値を超えた範囲を探し、その前後を補完
    extended_startvals = []
    extended_endvals = []

    # 閾値を超えた部分の開始と終了を取得
    startvals = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
    endvals = np.where(np.diff(above_threshold.astype(int)) == -1)[0] + 1

    # ピーク前後の部分も赤く表示するために、少し範囲を広げる
    extend_range = 20  # 20点分前後を拡張
    for start, end in zip(startvals, endvals):
        extended_startvals.append(max(0, start - extend_range))  # 開始点を拡張
        extended_endvals.append(
            min(len(signal) - 1, end + extend_range)
        )  # 終了点を拡張
    np.savetxt("extended_startvals_dp.csv", extended_startvals, delimiter=",", fmt="%d")
    np.savetxt("extended_endvals_dp.csv", extended_endvals, delimiter=",", fmt="%d")

    # 3Dプロット
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for k in range(len(startvals)):
        if extended_startvals[k] < len(V) and extended_endvals[k] < len(V):
            ax.plot(
                V[extended_startvals[k] : extended_endvals[k], 0],
                V[extended_startvals[k] : extended_endvals[k], 1],
                V[extended_startvals[k] : extended_endvals[k], 2],
                "r",
                linewidth=1.5,
            )
        else:
            print("エラー: インデックスが範囲外です。")

    for k in range(len(extended_startvals) - 1):
        if extended_endvals[k] < len(V) and extended_startvals[k + 1] < len(V):
            ax.plot(
                V[extended_endvals[k] : extended_startvals[k + 1], 0],
                V[extended_endvals[k] : extended_startvals[k + 1], 1],
                V[extended_endvals[k] : extended_startvals[k + 1], 2],
                color=[0.25, 0.25, 0.25],
                linewidth=1.5,
            )
        else:
            print("エラー: インデックスが範囲外です。")

    # プロットのその他の設定
    ax.set_xlabel("v_1")
    ax.set_ylabel("v_2")
    ax.set_zlabel("v_3")

    plt.tight_layout()
    fig.set_size_inches(10, 10)
    ax.view_init(elev=35, azim=45)
    plt.show()


def plot_time_series_forcing_active_double_pendulum(V, dt, r):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 9))
    tspan = np.arange(0, len(V) * dt, dt)

    # 信号
    L = np.arange(0, len(V))
    signal = V[L, r - 1] ** 2

    # 閾値を使った検出 (find_peaks の代わりに直接判定)
    threshold = 1.0e-5
    above_threshold = signal > threshold  # 閾値を超える部分はTrue

    # 閾値以下の部分も補完するためのロジック
    # 信号全体で閾値を超えた範囲を探し、その前後を補完
    extended_startvals = []
    extended_endvals = []

    # 閾値を超えた部分の開始と終了を取得
    startvals = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
    endvals = np.where(np.diff(above_threshold.astype(int)) == -1)[0] + 1

    # ピーク前後の部分も赤く表示するために、少し範囲を広げる
    extend_range = 20  # 20点分前後を拡張
    for start, end in zip(startvals, endvals):
        extended_startvals.append(max(0, start - extend_range))  # 開始点を拡張
        extended_endvals.append(
            min(len(signal) - 1, end + extend_range)
        )  # 終了点を拡張

    # グラフ描画

    # サブプロット1: v_1 の時系列データ
    ax1.plot(tspan[: len(V)], V[:, 0], "k")
    for start, end in zip(extended_startvals, extended_endvals):
        ax1.plot(tspan[start:end], V[start:end, 0], "r", linewidth=1.5)
    ax1.set_ylim([-0.0015, 0.0015])
    ax1.set_ylabel("v_1")

    # サブプロット2: v_{r} の時系列データ
    ax2.plot(tspan[: len(V)], V[:, r - 1], "k")
    for start, end in zip(extended_startvals, extended_endvals):
        ax2.plot(tspan[start:end], V[start:end, r - 1], "r", linewidth=1.5)
    ax2.set_ylabel(f"v_{r}")
    ax2.set_ylim([-0.008, 0.008])

    # サブプロット3: v_{r}^2 の時系列データ
    ax3.plot(tspan[: len(V)], signal, "k")
    for start, end in zip(extended_startvals, extended_endvals):
        ax3.plot(tspan[start:end], signal[start:end], "r", linewidth=1.5)
    ax3.set_xlabel("t")
    ax3.set_ylim([0, 7e-5])
    ax3.set_ylabel(f"v_{r}^2")

    # 共通の x 軸を設定
    ax1.get_shared_x_axes().join(ax1, ax2, ax3)
    ax1.set_xlim([20, 65])

    fig.tight_layout()
    plt.show()


def run_havok_analysis_duffing(data, m, r, dt, label):
    # Create Hankel matrix
    hankel_matrix = create_hankel_matrix(data, m)

    # Perform SVD
    U, S, Vt = truncated_svd(hankel_matrix, r)

    # Time-shifted matrices
    V_1 = Vt[:, :-1].T
    V_2 = Vt[:, 1:].T

    # Linear approximation A-hat
    A_1 = np.linalg.lstsq(V_2, V_1, rcond=None)[0]
    result_matrix = create_difference_matrix(A_1.shape[0], A_1) / dt

    # Create state-space model
    sys_new = create_linear_model(result_matrix, r)

    # Simulate the system
    V = Vt.T
    t_span, y = simulate_system(
        sys_new,
        V[:, r - 1],
        np.arange(0, len(V) * dt, dt),
        V[0, : r - 1],
    )

    # Plot results
    plot_attractor_forcing_active_duffing(V, r, dt)
    plot_time_series_forcing_active_duffing(V, dt, r)


def plot_attractor_forcing_active_duffing(V, r, dt):
    L = np.arange(0, len(V) // 2 - 1)
    inds = V[L, r - 1] ** 2 > 1.0e-6
    L = L[inds]

    startvals = []
    endvals = []
    start = 100
    numhits = 500

    for k in range(numhits):
        startvals.append(start)
        endmax = start + 100
        interval = np.arange(start, endmax)
        hits = np.where(inds[interval])[0]
        if start >= len(inds):
            print(f"エラー: start ({start}) が inds の範囲外です。")
        if endmax >= len(inds):
            print(f"エラー: endmax ({endmax}) が inds の範囲外です。")

        if hits.size > 0:
            endval = start + hits[-1]
            endvals.append(endval)
            newhit = np.where(inds[endval + 1 :])[0]
            if newhit.size == 0:
                print(
                    f"newhit が見つかりませんでした。endval = {endval}, start を更新できません。"
                )

            if newhit.size > 0:
                start = endval + newhit[0]
            else:
                print("新しいヒットが見つかりませんでした。ループを終了します。")
        else:
            print("ヒットが見つかりませんでした。ループを終了します。")

    print("startvals, endvals = :", len(startvals), len(endvals))

    # プロット
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # プロット前にサイズを確認
    assert len(startvals) == len(
        endvals
    ), "startvals と endvals のサイズが一致しません。"

    for k in range(numhits):
        if k < len(startvals) and k < len(endvals):
            print(f"k = {k+1}, startval = {startvals[k]}, endval = {endvals[k]}")
            if startvals[k] < len(V) and endvals[k] < len(V):
                ax.plot(
                    V[startvals[k] : endvals[k], 0],
                    V[startvals[k] : endvals[k], 1],
                    V[startvals[k] : endvals[k], 2],
                    "r",
                    linewidth=1.5,
                )
            else:
                print("エラー: インデックスが範囲外です。")
        else:
            print("エラー: 配列のサイズが不足しています。")

    for k in range(numhits - 1):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax.plot(
                    V[endvals[k] : startvals[k + 1], 0],
                    V[endvals[k] : startvals[k + 1], 1],
                    V[endvals[k] : startvals[k + 1], 2],
                    color=[0.25, 0.25, 0.25],
                    linewidth=1.5,
                )
            else:
                print("エラー: インデックスが範囲外です。")
        else:
            print("エラー: 配列のサイズが不足しています。")

    # プロットのその他の設定
    ax.set_xlabel("v_1")
    ax.set_ylabel("v_2")
    ax.set_zlabel("v_3")

    plt.tight_layout()
    fig.set_size_inches(10, 10)
    ax.view_init(elev=20, azim=120)
    plt.show()


def plot_time_series_forcing_active_duffing(V, dt, r):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 9))
    tspan = np.arange(0, len(V) * dt, dt)
    L = np.arange(0, len(V) // 2 - 1)
    inds = V[L, r - 1] ** 2 > 1.0e-6
    L = L[inds]

    startvals = []
    endvals = []
    start = 100
    numhits = 500

    for k in range(numhits):
        startvals.append(start)
        endmax = start + 100
        interval = np.arange(start, endmax)
        hits = np.where(inds[interval])[0]

        if hits.size > 0:
            endval = start + hits[-1]
            endvals.append(endval)
            newhit = np.where(inds[endval + 1 :])[0]

            if newhit.size > 0:
                start = endval + newhit[0]
            else:
                print("新しいヒットが見つかりませんでした。ループを終了します。")
        else:
            print("ヒットが見つかりませんでした。ループを終了します。")

    # サブプロット1: v_1 の時系列データ
    ax1.plot(tspan[: len(V)], V[:, 0], "k")
    for k in range(numhits):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax1.plot(
                    tspan[startvals[k] : endvals[k]],
                    V[startvals[k] : endvals[k], 0],
                    "r",
                    linewidth=1.5,
                )
            else:
                print("エラー：インデックスが範囲外")
        else:
            print("エラー：配列のサイズが不足")

    for k in range(numhits - 1):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax1.plot(
                    tspan[endvals[k] : startvals[k + 1]],
                    V[endvals[k] : startvals[k + 1], 0],
                    color=[0.25, 0.25, 0.25],
                    linewidth=1.5,
                )
            else:
                print("エラー：インデックスが範囲外")
        else:
            print("エラー：配列のサイズが不足")

    ax1.set_ylim([-0.0015, 0.0015])
    ax1.set_ylabel("v_1")

    # サブプロット2: v_{r} の時系列データ
    ax2.plot(tspan[: len(V)], V[:, r - 1], "k")
    for k in range(numhits):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax2.plot(
                    tspan[startvals[k] : endvals[k]],
                    V[startvals[k] : endvals[k], r - 1],
                    "r",
                    linewidth=1.5,
                )
            else:
                print("エラー：インデックスが範囲外")
        else:
            print("エラー：配列のサイズが不足")

    for k in range(numhits - 1):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax2.plot(
                    tspan[endvals[k] : startvals[k + 1]],
                    V[endvals[k] : startvals[k + 1], r - 1],
                    color=[0.25, 0.25, 0.25],
                    linewidth=1.5,
                )
            else:
                print("エラー：インデックスが範囲外")
        else:
            print("エラー：配列のサイズが不足")

    ax2.set_ylabel(f"v_{r}")

    # サブプロット3: v_{r}^2 の時系列データ
    ax3.plot(tspan[startvals[0]], V[startvals[0], r - 1], "r")
    ax3.plot(tspan[endvals[0]], V[startvals[1], r - 1], color=[0.25, 0.25, 0.25])
    for k in range(numhits):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax3.plot(
                    tspan[startvals[k] : endvals[k]],
                    V[startvals[k] : endvals[k], r - 1] ** 2,
                    "r",
                    linewidth=1.5,
                )
            else:
                print("エラー：インデックスが範囲外")
        else:
            print("エラー：配列のサイズが不足")

    for k in range(numhits - 1):
        if k < len(endvals) and k + 1 < len(startvals):
            print(f"k = {k+1}, endval = {endvals[k]}, startval = {startvals[k+1]}")
            if endvals[k] < len(V) and startvals[k + 1] < len(V):
                ax3.plot(
                    tspan[endvals[k] : startvals[k + 1]],
                    V[endvals[k] : startvals[k + 1], r - 1] ** 2,
                    color=[0.25, 0.25, 0.25],
                    linewidth=1.5,
                )
            else:
                print("エラー：インデックスが範囲外")
        else:
            print("エラー：配列のサイズが不足")

    ax3.set_xlabel("t")
    ax3.set_ylim([-0.000001, 0.00001])
    ax3.set_ylabel(f"v_{r}^2")

    # 共通の設定
    ax3.legend(["Forcing Active", "Forcing Inactive"])
    ax1.get_shared_x_axes().join(ax1, ax2, ax3)
    ax1.set_xlim([500, 800])

    # グラフのサイズや出力設定
    fig.tight_layout()
    plt.show()


# Main processing
def main():
    # Load data
    csv_files = ["duffing_x.csv", "dp_x1.csv", "lorenz_x.csv"]
    data = load_csv_data(csv_files)

    # Duffing System
    run_havok_analysis_duffing(
        data[0], m=50, r=8, dt=2.0 * np.pi / 100, label="Duffing"
    )

    # Double Pendulum System
    run_havok_analysis_double_pendulum(
        data[1], m=100, r=5, dt=0.001, label="Double Pendulum"
    )

    # Lorenz System
    run_havok_analysis(data[2], m=100, r=11, dt=0.001, label="Lorenz")


if __name__ == "__main__":
    main()
