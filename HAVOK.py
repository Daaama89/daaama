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
    sys_new = StateSpace(A_dp, B_dp, np.eye(r - 1), np.zeros((r - 1, 1)))
    return sys_new


# Simulate the linear system
def simulate_system(sys, input_data, t, initial_conditions):
    tspan, y, _ = lsim(sys, input_data, t, initial_conditions)
    return tspan, y


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


# Main processing
def main():
    # Load data
    csv_files = ["U.csv"] # <- Please change it for your preffer csv_file
    data = load_csv_data(csv_files)
    data = data[0]  # since load_csv_data returns a list of arrays

    # For debug
    # print(f"Loaded data length: {len(data)}")
    # print(f"Data sample: {data[:10]}")

    # Constants (please change it for your preffer values)
    dt = 0.001
    t = np.arange(0.0, 250, dt)
    m = 100
    r = 5

    # Create Hankel matrix
    hankel2 = create_hankel_matrix(data, m)
    # print('Hankel size',hankel2.shape) <- for debug

    # Perform SVD
    U, S, Vt = truncated_svd(hankel2, r)
    # print("Vt shape", Vt.shape)  # <- for debug

    # Time-shifted matrices
    V_1 = Vt[:, :-1].T
    V_2 = Vt[:, 1:].T
    # print("V1 and V2 shape", V_1.shape, V_2.shape) <- for debug

    # Linear approximation A-hat
    A_1 = np.linalg.lstsq(V_2, V_1, rcond=None)[0]
    result_matrix2 = create_difference_matrix(A_1.shape[0], A_1) / dt
    # print("Hankel_origin_A",result_matrix2.shape) <- for debug

    # Create state-space model
    sysNew2 = create_linear_model(result_matrix2, r)

    # Simulate the system
    V = Vt.T
    tspan, y = simulate_system(
        sysNew2, V[:, r - 1], np.arange(0, len(V) * dt, dt), V[0, : r - 1]
    )
    # print("tspan and y", len(tspan), y.shape) <- for debug

    fig, axs = plt.subplots(y.shape[1], 1, sharex=True, figsize=(7, 9))
    for i in range(y.shape[1]):
        axs[i].plot(
            t[50000:75000],
            V[50000:75000, i],
            "k",
            label="eigen-time-delay-coordinate",
        )
        axs[i].plot(
            t[50000:75000], y[50000:75000, i], "r--", label="HAVOK regression model"
        )
        axs[i].legend()
        axs[i].set(xlabel="t", ylabel=r"$v_{}$".format(i + 1))

    fig.show()
    # please change t_span, V_span, and HAVOK_model_span


if __name__ == "__main__":
    main()
