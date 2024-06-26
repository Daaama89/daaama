import matplotlib.animation as animation
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
t = np.arange(0.0, 15, dt)

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


U = -(M1 + M2) * G * L1 * np.cos(y[:, 0]) - M2 * G * L2 * np.cos(y[:, 2])

y_1 = y[:15000, 0]
y_2 = y[:15000, 2]


# アニメーションの作成
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(min(y_1), max(y_1))
ax.set_xlabel(r"$\theta_1[rad]$", fontsize=14)
ax.tick_params(axis="x", labelsize=14)
ax.set_ylim(min(U), max(U))
ax.set_ylabel(r"$U[\mathrm{J}]$", fontsize=14)
ax.tick_params(axis="y", labelsize=14)

(line1,) = ax.plot([], [], lw=2)


time_template = "time = %.1fs"
time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)


def init():
    line1.set_data([], [])
    time_text.set_text("")
    return line1, time_text


# アニメーション時間を15秒までに制限
animation_time_limit = 15.0
num_frames = int(animation_time_limit / dt)


def animate(i):
    line1.set_data(y_1[:i], U[:i])
    time_text.set_text(time_template % (i * dt))
    return line1, time_text


ani = animation.FuncAnimation(
    fig, animate, range(1, num_frames), interval=dt * 1000, blit=True, init_func=init
)

ani.save("animated_theta1.mp4", writer="ffmpeg", fps=int(1 / dt))

plt.show()
