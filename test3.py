import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 曲面の座標生成
def generate_surface():
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z

# 移動する点の座標生成
def generate_moving_point(theta):
    x = 0.5 * np.cos(theta)
    y = 0.5 * np.sin(theta)
    z = 0.5 * np.cos(2 * theta)
    return x, y, z

# 初期化関数
def init():
    ax.clear()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

# アニメーション関数
def update(frame):
    init()  # 初期化

    # 曲面をプロット
    ax.plot_surface(X, Y, Z, color='b', alpha=0.6, rstride=10, cstride=10)

    # 動く点の座標を計算
    theta = np.radians(frame)
    x, y, z = generate_moving_point(theta)

    # 動く点をプロット
    ax.scatter(x, y, z, c='red', s=50, marker='o', label='Moving Point')

# 3Dプロットのセットアップ
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = generate_surface()

# アニメーションの作成
animation = FuncAnimation(fig, update, frames=range(360), init_func=init, blit=False, repeat=False)

plt.show()