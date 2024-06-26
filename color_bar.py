import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import odeint

# サンプルの行列とベクトルを生成
matrix = np.random.randn(9, 9)
vector = np.random.randn(9)

# カスタムカラーマップを定義（青→白→赤）
colors = [(0, 'blue'), (0.5, 'white'), (1, 'red')]
cmap = LinearSegmentedColormap.from_list('custom', colors)

x_ticks = np.arange(-0.5,matrix.shape[1],1)
y_ticks = np.arange(-0.5,matrix.shape[0],1)

# 行列のプロット
plt.subplot(1, 2, 1)
plt.imshow(matrix, cmap=cmap, vmin=matrix.min(), vmax=matrix.max())
cbar = plt.colorbar()
cbar.set_label("Value")
plt.title("Matrix")
plt.grid(True, color='black',linewidth=0.5)
plt.xticks(x_ticks,[])
plt.yticks(y_ticks,[])

# ベクトルのプロット
plt.subplot(1, 2, 2)
plt.imshow(vector.reshape(-1, 1), cmap=cmap, vmin=vector.min(), vmax=vector.max())
cbar = plt.colorbar()
cbar.set_label("Value")
plt.title("Vector")

plt.tight_layout()
plt.show()



# 行列 A (14x14)
A = np.random.rand(14, 14)

# ベクトル B (14x1)
B = np.random.rand(14, 1)

# ダミー関数（微分方程式の右辺）
def dV_dt(V, t):
    dV = np.dot(A, V) + B.flatten()
    return dV

# 初期値 V0 (14x1)
V0 = np.zeros(14)

# 時間の範囲
t = np.linspace(0, 10, 100)  # 0から10まで100点で離散化

# 微分方程式を解く
result = odeint(dV_dt, V0, t)

# 結果を表示
print("V(t):", result)





# 仮に15×15の行列を生成（実際のデータに置き換える必要があります）
matrix_15x15 = np.random.rand(15, 15)
print("original",matrix_15x15)

# 15列目における15行目以外の要素を抜き出す
column_15_except_row_15 = matrix_15x15[:-1, 14]

# 結果を表示
print("B",column_15_except_row_15)