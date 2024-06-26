import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.signal import StateSpace, lsim


# Load CSV data
def load_csv_data(filenames, delimiter=",", dtype=float, encoding="ms932"):
    return [
        np.genfromtxt(file, delimiter=delimiter, dtype=dtype, encoding=encoding)
        for file in filenames
    ]


# Create a Hankel matrix
def create_hankel_matrix(data, m):
    k = len(data)
    n = k - m + 1
    hankel_matrix = np.column_stack([data[i : i + m] for i in range(n)])
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


# Plot data with highlighted regions
def plot_with_highlights(
    t,
    data,
    highlight_indices,
    ylabel,
    xlabel="time [s]",
    title="",
    highlight_color="red",
):
    plt.plot(t, data, "k", label=ylabel, linewidth=1)
    for i in highlight_indices:
        start_index = max(0, i - 100)
        end_index = min(len(t), i + 100)
        plt.fill_between(
            t[start_index:end_index],
            data[start_index:end_index],
            color=highlight_color,
            alpha=0.5,
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower left")
    plt.show()


# Highlight regions of forcing active
def highlight_forcing_active(V, threshold):
    return np.where(np.abs(V[:, 2]) > threshold)[0]


# Segment time indices
def segment_indices(t, indices, segments):
    segmented_indices = []
    for start, end in segments:
        segmented_indices.append(indices[(start <= t[indices]) & (t[indices] <= end)])
    return segmented_indices


# Main processing
def main():
    # Load data
    csv_files = [
        "x1.csv",
        "U.csv",
        "potential_time.csv",
        "peak_theta1.csv",
        "peak_theta2.csv",
        "all_peak_.csv",
        "local_minimum.csv",
    ]
    data = load_csv_data(csv_files)

    data2_U, data3_Utime, data4_peak1, data5_peak2, data6_peak12, data7_minimum = data[
        1:
    ]

    # Constants
    dt = 0.001
    t = np.arange(0.0, 250, dt)
    m2 = 100
    r = 5

    # Create Hankel matrix
    hankel2 = create_hankel_matrix(data2_U, m2)

    # Perform SVD
    U, S, V = truncated_svd(hankel2, r)

    # Time-shifted matrices
    V_1 = V[:, :-1].T
    V_2 = V[:, 1:].T

    # Linear approximation A-hat
    A_1 = np.linalg.lstsq(V_2, V_1, rcond=None)[0]
    result_matrix2 = create_difference_matrix(A_1.shape[0], A_1) / dt
    np.savetxt("HAVOK_original_A(U).txt", result_matrix2, fmt="%.18f", delimiter=",")

    # Create state-space model
    sysNew2 = create_linear_model(result_matrix2, r)

    # Simulate the system
    tspan2, y2 = simulate_system(
        sysNew2, V[:, r - 1], np.arange(0, len(V) * dt, dt), V[0, : r - 1]
    )

    # Highlight forcing active regions
    threshold2 = 0.003
    highlight_indices2 = highlight_forcing_active(V, threshold2)

    # Plot results with highlights
    plot_with_highlights(t[: -m2 + 1], V[4, :], highlight_indices2, ylabel="vr")

    # Further segment and plot data as required
    segments = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250)]
    segmented_indices = segment_indices(t, highlight_indices2, segments)

    for idx_segment in segmented_indices:
        plot_with_highlights(t, V[:, 4], idx_segment, ylabel="vr")

    # Count peaks and active points
    count_all = len(data6_peak12)
    count_active = sum(
        any((j - 1000 <= i < j) for j in highlight_indices2) for i in data6_peak12
    )
    count_peak1 = len(data4_peak1)
    count_active_peak1 = sum(
        any((j - 1000 <= i <= j + 1000) for j in highlight_indices2)
        for i in data4_peak1
    )
    count_peak2 = len(data5_peak2)
    count_active_peak2 = sum(
        any((j - 1000 <= i <= j + 1000) for j in highlight_indices2)
        for i in data5_peak2
    )
    count_minimum = len(data7_minimum)
    count_active_minimum = sum(
        any((j - 300 <= i < j + 300) for j in highlight_indices2) for i in data7_minimum
    )

    print("count_all", count_all)
    print("count_active", count_active)
    print("count_peak1", count_peak1)
    print("count_active_peak1", count_active_peak1)
    print("count_peak2", count_peak2)
    print("count_active_peak2", count_active_peak2)
    print("count_minimum", count_minimum)
    print("count_active_minimum", count_active_minimum)

    # Save forcing active data
    np.savetxt("forcing_active.csv", highlight_indices2, delimiter=",", fmt="%d")


if __name__ == "__main__":
    main()
