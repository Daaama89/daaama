import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint


def f(var, t, alpha, beta, delta, gamma, omega, phi):
    ''' Duffing oscillator '''

    v = var[1]
    a = - alpha * var[0] - beta * var[0] ** 3 - delta * var[1],
    + gamma * np.cos(omega * t + phi)

    return np.array([v, a])


def plot(t, x, v):
    ''' プロット '''

    # フォントの種類とサイズを設定する。
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'

    # 目盛を内側にする。
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # グラフの上下左右に目盛線を付ける。
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(221)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax2 = fig.add_subplot(223)
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')
    ax3 = fig.add_subplot(122)
    ax3.yaxis.set_ticks_position('both')
    ax3.xaxis.set_ticks_position('both')

    # 軸のラベルを設定する。
    ax1.set_xlabel('t[s]')
    ax1.set_ylabel('x[m]')
    ax2.set_xlabel('t[s]')
    ax2.set_ylabel('v[m/s]')
    ax3.set_xlabel('x[m]')
    ax3.set_ylabel('v[m/s]')

    # スケールの設定をする。
    ax1.set_ylim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)

    # データプロットの準備とともに、ラベルと線の太さ、凡例の設置を行う。
    ax1.plot(t, x, label='Displacement', lw=1, color='red')
    ax2.plot(t, v, label='Velocity', lw=1, color='red')
    ax3.scatter(x, v, label='Phase plane', lw=1, color='red', s=1)

    # レイアウト設定
    fig.tight_layout()

    # グラフを表示する。
    plt.show()
    plt.close()


if __name__ == '__main__':
    ''' メイン処理 '''

    # ダフィング振動子のパラメータ
    alpha = -1.0
    beta = 1.0
    delta = 0.2
    gamma = 0.3
    omega = 1.0
    phi = 0

    # 解析時間
    t = np.arange(0, 10000, (2 * np.pi) / omega)

    # 初期条件
    var = [0.0, 0.0]

    # 微分方程式の近似解法（過渡応答）
    var = odeint(f, var, t, args=(alpha, beta, delta, gamma, omega, phi))
    x, v = var.T[0], var.T[1]

    # プロット
    plot(t, x, v)
