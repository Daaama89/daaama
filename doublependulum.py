import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# パラメータ
g = 9.81  # 重力加速度 (m/s^2)
L1 = 1.0  # 一本目の振り子の長さ (m)
L2 = 1.0  # 二本目の振り子の長さ (m)
m1 = 1.0
m2 = 1.0

# 初期条件
theta1_0 = np.pi / 2  # 一本目の振り子の初期角度
theta2_0 = np.pi / 2  # 二本目の振り子の初期角度
omega1_0 = 0.0  # 一本目の振り子の初期角速度
omega2_0 = 0.0  # 二本目の振り子の初期角速度


def double_pendulum(t, state):
    theta1, theta2, omega1, omega2 = state
    
    # 振り子の角度の時間微分
    delta_theta1 = omega1
    delta_theta2 = omega2
    
    # 振り子の角速度の時間微分
    delta_omega1 = (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) - 2 * np.sin(theta1 - theta2) * m2 * (omega2 ** 2 * L2 + omega1 ** 2 * L1 * np.cos(theta1 - theta2))) / (L1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
    delta_omega2 = 2 * np.sin(theta1 - theta2) * (omega1 ** 2 * L1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + omega2 ** 2 * L2 * m2 * np.cos(theta1 - theta2)) / (L2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
    
    return [delta_theta1, delta_theta2, delta_omega1, delta_omega2]

# 時間の設定
t_span = (0, 100)  # シミュレーションの時間範囲
t_eval = np.arange(0, 100, 0.001)  # 時間の離散化

# 初期条件
initial_state = [theta1_0, theta2_0, omega1_0, omega2_0]

# 振り子のシミュレーション
sol = solve_ivp(double_pendulum, t_span, initial_state, t_eval=t_eval)

# ポアンカレ写像を作成
theta1_values = sol.y[0]
theta2_values = sol.y[1]

# 図のプロット
plt.scatter(theta1_values, theta2_values, s=1)
plt.xlabel('Theta1 (radians)')
plt.ylabel('Theta2 (radians)')
plt.title('Double Pendulum Poincaré Map')
plt.show()
