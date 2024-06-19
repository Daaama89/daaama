import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import scipy.integrate as integrate
from numpy import cos, sin

dt = 0.002

t_end_train = 20
t_end_test = 40

t_train = np.arange(0, t_end_train, dt)
t_test = np.arange(0, t_end_test, dt)

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


# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 135.0
w1 = 0.0
th2 = 135.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t_train)
DP_model = np.stack((y[:, 0], y[:, 1], y[:, 2], y[:, 3]), axis=-1)

model_dp = ps.SINDy(feature_names=["$theta_1$", "$omega_1$", "$theta_2$", "$omega_2$"])
model_dp.fit(DP_model, t=dt)

print("test_model:")
model_dp.print()

z = integrate.odeint(derivs, state, t_test)
DP_test = np.stack((z[:, 0], z[:, 1], z[:, 2], z[:, 3]), axis=-1)
print("model score: %f" % model_dp.score(DP_test, t=dt))

DP_dot_test_predict = model_dp.predict(DP_test)
DP_dot_test_computed = model_dp.differentiate(DP_test, t=dt)

fig, axs = plt.subplots(y.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(y.shape[1]):
    axs[i].plot(t_test, DP_dot_test_computed[:, i], "k", label="numerical derivative")
    axs[i].plot(t_test, DP_dot_test_predict[:, i], "r--", label="model prediction")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))

fig.show()
