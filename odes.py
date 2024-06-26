import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# A行列とBベクトルの定義
A = np.random.rand(14, 14)  # 仮の14x14のランダム行列
B = np.random.rand(14)      # 仮の14次元のランダムベクトル

# 微分方程式の定義
def system(variables, t):
    v = variables[:14]  # v_1 から v_14
    v15 = variables[14]  # v_15

    dVdt = np.dot(A, v) + B * v15
    return np.append(dVdt, v15)  # 0はv_15の微分を0に設定

# 初期条件の設定
initial_conditions = np.random.rand(15)

# 時間の範囲を指定
time = np.linspace(0, 10, 1000)

# 微分方程式を解く
solution = odeint(system, initial_conditions, time)

# 結果のプロット
plt.figure(figsize=(10, 6))
for i in range(15):
    plt.plot(time, solution[:, i], label=f"v_{i+1}")

plt.title("Solution Trajectories")
plt.xlabel("Time")
plt.ylabel("Variable Values")
plt.legend()
plt.show()