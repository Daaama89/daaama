import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 元のデータ（適当な例としてsin波）
t_original = np.linspace(0, 10, 100)
X_original = np.vstack([np.sin(t_original), np.cos(t_original)]).T

# HAVOK解析によって得られた係数行列A（適当な例としてランダム行列）
A = np.random.rand(2, 2)

# 初期条件
initial_conditions = X_original[0]

# 時間の範囲
t = np.linspace(0, 10, 100)

# 線形微分方程式の右辺を定義
def model(x, t):
    return np.dot(A, x)

# 数値的に解く（HAVOK解析の解）
solution_havok = odeint(model, initial_conditions, t)

# 元のデータの時間遅れ座標系の軌道
plt.plot(t_original, X_original[:, 0], label='Original Data (Dim 1)')
plt.plot(t_original, X_original[:, 1], label='Original Data (Dim 2)')

# HAVOK解析による線形微分方程式の解の軌道
plt.plot(t, solution_havok[:, 0], label='HAVOK Solution (Dim 1)', linestyle='solid')
plt.plot(t, solution_havok[:, 1], label='HAVOK Solution (Dim 2)', linestyle='solid')

plt.xlabel('Time')
plt.ylabel('Dimension Value')
plt.legend()
plt.show()



# HAVOK解析によって得られたAとB
A = np.array([[0.5, -0.2],
              [0.1, -0.4]])

B = np.array([[0.3],
              [0.2]])

# 線形微分方程式 dx/dt = Ax + Bu を定義
def linear_system(t, x):
    u = np.sin(t)  # 外部入力（例：sin関数）
    return np.dot(A, x) + np.dot(B, u)

# 初期条件
x0 = np.array([1.0, 0.5])

# 時間の範囲を指定
t_span = (0, 5)

# 線形微分方程式を解く
sol = solve_ivp(linear_system, t_span, x0, dense_output=True)

# 解を評価するための時間点を指定
t_eval = np.linspace(0, 5, 100)

# 解を評価
x_eval = sol.sol(t_eval)

# 結果のプロット
plt.plot(t_eval, x_eval[0], label='x1')
plt.plot(t_eval, x_eval[1], label='x2')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.legend()
plt.show()
