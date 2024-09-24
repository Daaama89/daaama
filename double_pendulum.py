import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from numpy import cos, sin

G = 9.81  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


# define EOM
def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2) * L1 - M2 * L1 * cos(del_) * cos(del_)
    dydx[1] = (
        M2 * L1 * state[1] * state[1] * sin(del_) * cos(del_)
        + M2 * G * sin(state[2]) * cos(del_)
        + M2 * L2 * state[3] * state[3] * sin(del_)
        - (M1 + M2) * G * sin(state[0])
    ) / den1

    dydx[2] = state[3]

    den2 = (L2 / L1) * den1
    dydx[3] = (
        -M2 * L2 * state[3] * state[3] * sin(del_) * cos(del_)
        + (M1 + M2) * G * sin(state[0]) * cos(del_)
        - (M1 + M2) * L1 * state[1] * state[1] * sin(del_)
        - (M1 + M2) * G * sin(state[2])
    ) / den2

    return dydx


# create a time array at 0.001 second steps
dt = 0.001
t = np.arange(0.0, 1500, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 135.0
w1 = 0.0
th2 = 135.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

# y[:,0] = theta1
# y[:,1] = omega1
# y[:,2] = theta2
# y[:,3] = omega2

U = -(M1 + M2) * G * L1 * np.cos(y[:, 0]) - M2 * G * L2 * np.cos(y[:, 2])

x_1 = L1 * sin(y[:, 0])
x_2 = L2 * sin(y[:, 2]) + L1 * sin(y[:, 0])
y_1 = -L1 * cos(y[:, 0])
y_2 = -L2 * cos(y[:, 2]) - L1 * cos(y[:, 0])


data_list1 = 1 / 2 * (M1 + M2) * L1**2 * y[:, 1] ** 2
data_list2 = 1 / 2 * M2 * L2**2 * y[:, 3] ** 2
data_list3 = L2 * L1 * y[:, 1] * y[:, 3] * cos(y[:, 1] - y[:, 3])
data_list4 = -(M1 + M2) * G * L1 * cos(y[:, 1])
data_list5 = -M2 * G * L2 * cos(y[:, 3])


# 各データをリストにまとめる
all_data = [data_list1, data_list2, data_list3, data_list4, data_list5]

# 各データを5分割
split_data_list = [np.array_split(data, 30) for data in all_data]

# for set_num in range(6):
# fig, axes = plt.subplots(5, 1, figsize=(10, 15))  # 1つのセットで5つのグラフを表示

# 5つの区間を1セットとして表示
# for i in range(5):
# current_segment = set_num * 5 + i  # 表示するセグメントの番号
# time_segment = t[
# current_segment * (len(t) // 30) : (current_segment + 1) * (len(t) // 30)
# ]  ## 対応する時間軸を分割

# for j, split_data in enumerate(split_data_list):
# axes[i].plot(
# time_segment, split_data[current_segment], label=f"Data {j+1}"
# )  ## 指定された時間軸を使用してプロット
# axes[i].set_xlabel("Time (seconds)")
# axes[i].set_ylabel(f"Segment {current_segment + 1}")
# axes[i].legend()

# レイアウトの調整
# plt.tight_layout()
# plt.show()


# position_time_series
np.savetxt("dp_x1_135.csv", x_1, delimiter=",", fmt="%.18f")
np.savetxt("dp_x2_135.csv", x_2, delimiter=",", fmt="%.18f")
np.savetxt("dp_y1_135.csv", y_1, delimiter=",", fmt="%.18f")
np.savetxt("dp_y2_135.csv", y_2, delimiter=",", fmt="%.18f")

# Mechanical_Energy_time_series
np.savetxt(
    "dp_energy_first_term_135.csv",
    1 / 2 * (M1 + M2) * L1**2 * y[:, 1] ** 2,
    delimiter=",",
    fmt="%.18f",
)
np.savetxt(
    "dp_energy_second_term_135.csv",
    1 / 2 * M2 * L2**2 * y[:, 3] ** 2,
    delimiter=",",
    fmt="%.18f",
)
np.savetxt(
    "dp_energy_third_term_135.csv",
    L2 * L1 * y[:, 1] * y[:, 3] * cos(y[:, 1] - y[:, 3]),
    delimiter=",",
    fmt="%.18f",
)
np.savetxt(
    "dp_energy_fourth_term_135.csv",
    -(M1 + M2) * G * L1 * cos(y[:, 1]),
    delimiter=",",
    fmt="%.18f",
)
np.savetxt(
    "dp_energy_fifth_term_135.csv",
    -M2 * G * L2 * cos(y[:, 3]),
    delimiter=",",
    fmt="%.18f",
)

np.savetxt("dp_theta1_135.csv", y[:, 0], delimiter=",", fmt="%.18f")
np.savetxt("dp_omega1_135.csv", y[:, 1], delimiter=",", fmt="%.18f")
np.savetxt("dp_theta2_135.csv", y[:, 2], delimiter=",", fmt="%.18f")
np.savetxt("dp_omega2_135.csv", y[:, 3], delimiter=",", fmt="%.18f")


T_1 = y[:15000, 0]
T_2 = y[:15000, 2]
T_1_grid, T_2_grid = np.meshgrid(T_1, T_2)
U_potential = -(M1 + M2) * G * L1 * np.cos(T_1_grid) - M2 * G * L2 * np.cos(T_2_grid)

# 等高線プロット
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(T_1_grid, T_2_grid, U_potential, cmap="Reds")

# 軸ラベル
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.set_zlabel("U")
ax.view_init(elev=35, azim=20)

# グラフを表示
plt.tight_layout()
plt.show()
