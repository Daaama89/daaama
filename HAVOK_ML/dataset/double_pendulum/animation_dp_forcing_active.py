import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

# 各変数が別々のCSVファイルにあると仮定
x_data = pd.read_csv("dp_V1.csv", header=None)  # x座標のデータ
y_data = pd.read_csv("dp_V2.csv", header=None)  # y座標のデータ
z_data = pd.read_csv("dp_V3.csv", header=None)  # z座標のデータ

# データを配列に変換
x = x_data[0].values
y = y_data[0].values
z = z_data[0].values

# 3Dプロットのセットアップ
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 初期化
(line,) = ax.plot([], [], [], lw=2)
(point,) = ax.plot([], [], [], "ro")

# 軸の範囲を設定
ax.set_xlim((min(x), max(x)))
ax.set_ylim((min(y), max(y)))
ax.set_zlim((min(z), max(z)))
ax.view_init(elev=35, azim=45)


# アップデート関数
def update(num):
    line.set_data(x[:num], y[:num])
    line.set_3d_properties(z[:num])

    point.set_data(x[num : num + 1], y[num : num + 1])
    point.set_3d_properties(z[num : num + 1])
    return line, point


# アニメーションの作成
ani = FuncAnimation(fig, update, frames=20000, interval=5, blit=True)

# アニメーションの表示
plt.show()
ani.save("animation_forcing.mp4", writer="ffmpeg")
