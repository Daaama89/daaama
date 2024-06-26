import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.signal import StateSpace, lsim

# import csv file
csv_file1 = "x1.csv"
data1_x1 = np.genfromtxt(
    csv_file1,
    delimiter=",",
    dtype=float,
    encoding="ms932",
)

csv_file2 = "U.csv"
data2_U = np.genfromtxt(
    csv_file2,
    delimiter=",",
    dtype=float,
    encoding="ms932",
)

csv_file3 = "potential_time.csv"
data3_Utime = np.genfromtxt(
    csv_file3,
    delimiter=",",
    dtype=int,
    encoding="ms932",
)

csv_file4 = "peak_theta1.csv"
data4_peak1 = np.genfromtxt(
    csv_file4,
    delimiter=",",
    dtype=int,
    encoding="ms932",
)

csv_file5 = "peak_theta2.csv"
data5_peak2 = np.genfromtxt(
    csv_file5,
    delimiter=",",
    dtype=int,
    encoding="ms932",
)

csv_file6 = "all_peak_.csv"
data6_peak12 = np.genfromtxt(
    csv_file6,
    delimiter=",",
    dtype=int,
    encoding="ms932",
)

csv_file7 = "local_minimum.csv"
data7_minimum = np.genfromtxt(
    csv_file7,
    delimiter=",",
    dtype=int,
    encoding="ms932",
)


# 2. HAVOK analysis(U)

# 2.1 make Hankel matrix
k2 = len(data2_U)
m2 = 100
n2 = k2 - m2 + 1
dt = 0.001
t = np.arange(0.0, 250, dt)
r = 5

a2 = np.column_stack([data2_U[i : i + m2] for i in range(0, n2)])
hankel2 = np.array(a2)

# 1.2 SVD
U2, S2, V2 = sp.linalg.svd(hankel2, full_matrices=0)


# 1.3 make the linear model (use DMD)

# (1) low rank r
Ur2 = U2[:, :r]
Sr2 = np.diag(S2[:r])
Vr2 = V2[:r, :]

# (2) make time-shifted matrices
V_1_2 = Vr2[:, :-1].T
V_2_2 = Vr2[:, 1:].T

# (3) make linear approximation A-hat
A_1_2 = np.linalg.lstsq(V_2_2, V_1_2, rcond=None)[0]


# (4) make A
def create_difference_matrix(n, X):
    # make I
    identity_matrix = np.identity(n)

    # input A_1
    arbitrary_matrix = X

    # subtraction A_1 and I
    difference_matrix = arbitrary_matrix - identity_matrix[:n, :n]

    return difference_matrix


result_matrix2 = create_difference_matrix(A_1_2.shape[0], A_1_2)
result_matrix2 = result_matrix2 / dt
np.savetxt(
    "HAVOK_original_A(U).txt",
    result_matrix2,
    fmt="%.18f",
    delimiter=",",
)

# (5) make linear model
A2_dp = result_matrix2[: (r - 1), : (r - 1)]
B2_dp = result_matrix2[:-1, (r - 1)]
np.savetxt("HAVOK_A(U).txt", A2_dp, fmt="%.18f", delimiter=",")
np.savetxt("HAVOK_B(U).txt", B2_dp, fmt="%.18f", delimiter=",")
A2_dp = np.array(A2_dp)
B2_dp = np.array(B2_dp)

# 3.2.1 original HAVOK(v1 and v5)
B2_dp = B2_dp.reshape(-1, 1)
sysNew2 = StateSpace(A2_dp, B2_dp, np.eye(r - 1), np.zeros((r - 1, 1)))

Vr2 = Vr2.T
tspan2, y2, _ = lsim(
    sysNew2, Vr2[:, (r - 1)], np.arange(0, len(Vr2) * dt, dt), Vr2[0, : (r - 1)]
)

V2 = V2.T


# figure
# forcing active

# parameter

threshold2 = 0.003
condition2 = (-1.0 * 10 ** (-5) < V2[:, 0]) & (V2[:, 0] < 1.0 * 10 ** (-5))
G = 9.81  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
U_threshold_down_up = -(M1 + M2) * G * L1 * np.cos(0) - M2 * G * L2 * np.cos(np.pi)
U_threshold_up_down = -(M1 + M2) * G * L1 * np.cos(np.pi) - M2 * G * L2 * np.cos(0)
highlight_indices_down_up = np.where(data2_U < U_threshold_down_up)[0]
highlight_indices_up_down = np.where(data2_U < U_threshold_up_down)[0]


# highlighted for forcing active
highlight_indices2 = np.where(np.abs(V2[:, 2]) > threshold2)[0]

plt.plot(t[: -m2 + 1], V2[:, 4], "k", label="vr", linewidth=1)
for i in highlight_indices2:
    start_index2 = max(0, i - 100)
    end_index2 = min(len(t), i + 100)
    plt.fill_between(
        t[start_index2:end_index2],
        V2[start_index2:end_index2, 4],
        color="red",
        alpha=0.5,
    )
plt.xlabel("time [s]")
plt.ylabel("vr")
plt.legend(loc="lower left")
plt.show()

plt.plot(t[: -m2 + 1], V2[:, 0], "k", label="v1", linewidth=1)
plt.scatter(
    t[highlight_indices2],
    V2[highlight_indices2, 0],
    color="red",
    label="forcing active",
    s=5,
)
plt.xlabel("time[s]")
plt.ylabel("v1")
plt.title("v1(original=U, threshold = 0.003) & t")
plt.legend(loc="lower left")
plt.show()

highlight_indices2_0_50s = highlight_indices2[t[highlight_indices2] <= 50]
highlight_indices2_50_100s = highlight_indices2[
    (50 <= t[highlight_indices2]) & (t[highlight_indices2] <= 100)
]
highlight_indices2_100_150s = highlight_indices2[
    (100 <= t[highlight_indices2]) & (t[highlight_indices2] <= 150)
]
highlight_indices2_150_200s = highlight_indices2[
    (150 <= t[highlight_indices2]) & (t[highlight_indices2] <= 200)
]
highlight_indices2_200_250s = highlight_indices2[
    (200 <= t[highlight_indices2]) & (t[highlight_indices2] <= 250)
]

Utimes_0_50s = data3_Utime[t[data3_Utime] <= 50]
Utimes_50_100s = data3_Utime[(50 <= t[data3_Utime]) & (t[data3_Utime] <= 100)]
Utimes_100_150s = data3_Utime[(100 <= t[data3_Utime]) & (t[data3_Utime] <= 150)]
Utimes_150_200s = data3_Utime[(150 <= t[data3_Utime]) & (t[data3_Utime] <= 200)]
Utimes_200_250s = data3_Utime[(200 <= t[data3_Utime]) & (t[data3_Utime] <= 250)]

peak1_0_50s = data4_peak1[t[data4_peak1] <= 50]
peak1_50_100s = data4_peak1[(50 <= t[data4_peak1]) & (t[data4_peak1] <= 100)]
peak1_100_150s = data4_peak1[(100 <= t[data4_peak1]) & (t[data4_peak1] <= 150)]
peak1_150_200s = data4_peak1[(150 <= t[data4_peak1]) & (t[data4_peak1] <= 200)]
peak1_200_250s = data4_peak1[(200 <= t[data4_peak1]) & (t[data4_peak1] <= 250)]

peak2_0_50s = data5_peak2[t[data5_peak2] <= 50]
peak2_50_100s = data5_peak2[(50 <= t[data5_peak2]) & (t[data5_peak2] <= 100)]
peak2_100_150s = data5_peak2[(100 <= t[data5_peak2]) & (t[data5_peak2] <= 150)]
peak2_150_200s = data5_peak2[(150 <= t[data5_peak2]) & (t[data5_peak2] <= 200)]
peak2_200_250s = data5_peak2[(200 <= t[data5_peak2]) & (t[data5_peak2] <= 250)]

peak12_0_50s = data6_peak12[t[data6_peak12] <= 50]
peak12_50_100s = data6_peak12[(50 <= t[data6_peak12]) & (t[data6_peak12] <= 100)]
peak12_100_150s = data6_peak12[(100 <= t[data6_peak12]) & (t[data6_peak12] <= 150)]
peak12_150_200s = data6_peak12[(150 <= t[data6_peak12]) & (t[data6_peak12] <= 200)]
peak12_200_250s = data6_peak12[(200 <= t[data6_peak12]) & (t[data6_peak12] <= 250)]


minimum_0_50s = data7_minimum[t[data7_minimum] <= 50]
minimum_50_100s = data7_minimum[(50 <= t[data7_minimum]) & (t[data7_minimum] <= 100)]
minimum_100_150s = data7_minimum[(100 <= t[data7_minimum]) & (t[data7_minimum] <= 150)]
minimum_150_200s = data7_minimum[(150 <= t[data7_minimum]) & (t[data7_minimum] <= 200)]
minimum_200_250s = data7_minimum[(200 <= t[data7_minimum]) & (t[data7_minimum] <= 250)]


plt.plot(t[:50000], V2[:50000, 4], "k", label="vr", linewidth=1)
for i in highlight_indices2_0_50s:
    start_index2 = max(0, i - 100)
    end_index2 = min(len(t), i + 100)
    plt.fill_between(
        t[start_index2:end_index2],
        V2[start_index2:end_index2, 4],
        color="red",
        alpha=0.5,
    )
plt.xlabel("time [s]", fontsize=18)
plt.ylabel("vr", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="lower left")
plt.show()

plt.plot(t[50000:100000], V2[50000:100000, 4], "k", label="vr", linewidth=1)
for i in highlight_indices2_50_100s:
    start_index2 = max(0, i - 100)
    end_index2 = min(len(t), i + 100)
    plt.fill_between(
        t[start_index2:end_index2],
        V2[start_index2:end_index2, 4],
        color="red",
        alpha=0.5,
    )
plt.xlabel("time [s]", fontsize=18)
plt.ylabel("vr", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="lower left")
plt.show()

plt.plot(t[100000:150000], V2[100000:150000, 4], "k", label="vr", linewidth=1)
for i in highlight_indices2_100_150s:
    start_index2 = max(0, i - 100)
    end_index2 = min(len(t), i + 100)
    plt.fill_between(
        t[start_index2:end_index2],
        V2[start_index2:end_index2, 4],
        color="red",
        alpha=0.5,
    )
plt.xlabel("time [s]", fontsize=18)
plt.ylabel("vr", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="lower left")
plt.show()

plt.plot(t[150000:200000], V2[150000:200000, 4], "k", label="vr", linewidth=1)
for i in highlight_indices2_150_200s:
    start_index2 = max(0, i - 100)
    end_index2 = min(len(t), i + 100)
    plt.fill_between(
        t[start_index2:end_index2],
        V2[start_index2:end_index2, 4],
        color="red",
        alpha=0.5,
    )
plt.xlabel("time [s]", fontsize=18)
plt.ylabel("vr", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="lower left")
plt.show()

plt.plot(
    t[200000 : 250000 - m2 + 1],
    V2[200000 : 250000 - m2 + 1, 4],
    "k",
    label="vr",
    linewidth=1,
)
for i in highlight_indices2_200_250s:
    start_index2 = max(0, i - 100)
    end_index2 = min(len(t), i + 100)
    plt.fill_between(
        t[start_index2:end_index2],
        V2[start_index2:end_index2, 4],
        color="red",
        alpha=0.5,
    )
plt.xlabel("time [s]", fontsize=18)
plt.ylabel("vr", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="lower left")
plt.show()


# potential

# local minimum
plt.plot(t[:50000], data2_U[:50000], "k", label="potential[J]", linewidth=1)
plt.scatter(
    t[highlight_indices2_0_50s],
    data2_U[highlight_indices2_0_50s],
    color="red",
    label="forcing active",
    s=5,
)
plt.scatter(
    t[minimum_0_50s],
    data2_U[minimum_0_50s],
    color="blue",
    label="equilibrium point",
    s=15,
)
plt.xlabel("time[s]", fontsize=18)
plt.ylabel("potential [J]", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="upper right")
plt.show()

plt.plot(t[50000:100000], data2_U[50000:100000], "k", label="potential[J]", linewidth=1)
plt.scatter(
    t[highlight_indices2_50_100s],
    data2_U[highlight_indices2_50_100s],
    color="red",
    label="forcing active",
    s=5,
)
plt.scatter(
    t[minimum_50_100s],
    data2_U[minimum_50_100s],
    color="blue",
    label="equilibrium point",
    s=15,
)
plt.xlabel("time[s]", fontsize=18)
plt.ylabel("potential [J]", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="upper right")
plt.show()

plt.plot(
    t[100000:150000], data2_U[100000:150000], "k", label="potential[J]", linewidth=1
)
plt.scatter(
    t[highlight_indices2_100_150s],
    data2_U[highlight_indices2_100_150s],
    color="red",
    label="forcing active",
    s=5,
)
plt.scatter(
    t[minimum_100_150s],
    data2_U[minimum_100_150s],
    color="blue",
    label="equilibrium point",
    s=15,
)
plt.xlabel("time[s]", fontsize=18)
plt.ylabel("potential [J]", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="upper right")
plt.show()

plt.plot(
    t[150000:200000], data2_U[150000:200000], "k", label="potential[J]", linewidth=1
)
plt.scatter(
    t[highlight_indices2_150_200s],
    data2_U[highlight_indices2_150_200s],
    color="red",
    label="forcing active",
    s=5,
)
plt.scatter(
    t[minimum_150_200s],
    data2_U[minimum_150_200s],
    color="blue",
    label="equilibrium point",
    s=15,
)
plt.xlabel("time[s]", fontsize=18)
plt.ylabel("potential [J]", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="upper right")
plt.show()

plt.plot(
    t[200000 : 250000 - m2 + 1],
    data2_U[200000 : 250000 - m2 + 1],
    "k",
    label="potential[J]",
    linewidth=1,
)
plt.scatter(
    t[highlight_indices2_200_250s],
    data2_U[highlight_indices2_200_250s],
    color="red",
    label="forcing active",
    s=5,
)
plt.scatter(
    t[minimum_200_250s],
    data2_U[minimum_200_250s],
    color="blue",
    label="equilibrium point",
    s=15,
)
plt.xlabel("time[s]", fontsize=18)
plt.ylabel("potential [J]", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="upper right")
plt.show()

# saddle
plt.plot(t[:50000], data2_U[:50000], "k", label="potential[J]", linewidth=1)
plt.scatter(
    t[highlight_indices2_0_50s],
    data2_U[highlight_indices2_0_50s],
    color="red",
    label="forcing active",
    s=5,
)
plt.scatter(t[peak12_0_50s], data2_U[peak12_0_50s], color="blue", label="saddle", s=15)
plt.xlabel("time[s]", fontsize=18)
plt.ylabel("potential [J]", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="upper right")
plt.show()

plt.plot(t[50000:100000], data2_U[50000:100000], "k", label="potential[J]", linewidth=1)
plt.scatter(
    t[highlight_indices2_50_100s],
    data2_U[highlight_indices2_50_100s],
    color="red",
    label="forcing active",
    s=5,
)
plt.scatter(
    t[peak12_50_100s], data2_U[peak12_50_100s], color="blue", label="saddle", s=15
)
plt.xlabel("time[s]", fontsize=18)
plt.ylabel("potential [J]", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="upper right")
plt.show()

plt.plot(
    t[100000:150000], data2_U[100000:150000], "k", label="potential[J]", linewidth=1
)
plt.scatter(
    t[highlight_indices2_100_150s],
    data2_U[highlight_indices2_100_150s],
    color="red",
    label="forcing active",
    s=5,
)
plt.scatter(
    t[peak12_100_150s], data2_U[peak12_100_150s], color="blue", label="saddle", s=15
)
plt.xlabel("time[s]", fontsize=18)
plt.ylabel("potential [J]", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="upper right")
plt.show()

plt.plot(
    t[150000:200000], data2_U[150000:200000], "k", label="potential[J]", linewidth=1
)
plt.scatter(
    t[highlight_indices2_150_200s],
    data2_U[highlight_indices2_150_200s],
    color="red",
    label="forcing active",
    s=5,
)
plt.scatter(
    t[peak12_150_200s], data2_U[peak12_150_200s], color="blue", label="saddle", s=15
)
plt.xlabel("time[s]", fontsize=18)
plt.ylabel("potential [J]", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="upper right")
plt.show()

plt.plot(
    t[200000 : 250000 - m2 + 1],
    data2_U[200000 : 250000 - m2 + 1],
    "k",
    label="potential[J]",
    linewidth=1,
)
plt.scatter(
    t[highlight_indices2_200_250s],
    data2_U[highlight_indices2_200_250s],
    color="red",
    label="forcing active",
    s=5,
)
plt.scatter(
    t[peak12_200_250s], data2_U[peak12_200_250s], color="blue", label="saddle", s=15
)
plt.xlabel("time[s]", fontsize=18)
plt.ylabel("potential [J]", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc="upper right")
plt.show()


count_all = len(data6_peak12)
print("count_all", count_all)

count_active = sum(
    any((j - 1000 <= i < j) for j in highlight_indices2) for i in data6_peak12
)
print("count_active", count_active)

count_peak1 = len(data4_peak1)
count_active_peak1 = sum(
    any((j - 1000 <= i <= j + 1000) for j in highlight_indices2) for i in data4_peak1
)
print("count_peak1", count_peak1)
print("count_active_peak1", count_active_peak1)

count_peak2 = len(data5_peak2)
count_active_peak2 = sum(
    any((j - 1000 <= i <= j + 1000) for j in highlight_indices2) for i in data5_peak2
)
print("count_peak1", count_peak2)
print("count_active_peak1", count_active_peak2)

count_minimum = len(data7_minimum)
print("count_minimum", count_minimum)

count_active2 = sum(
    any((j - 300 <= i < j + 300) for j in highlight_indices2) for i in data7_minimum
)
print("count_active2", count_active2)


data9 = highlight_indices2
np.savetxt("forcing_active.csv", data9, delimiter=",", fmt="%d")
