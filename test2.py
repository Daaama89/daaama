from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
from scipy.optimize import minimize
from scipy.signal import argrelextrema
from scipy.interpolate import griddata

G  = 9.81  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg

#define EOM
def derivs(state, t):
    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
               M2*G*sin(state[2])*cos(del_) +
               M2*L2*state[3]*state[3]*sin(del_) -
               (M1 + M2)*G*sin(state[0]))/den1

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
               (M1 + M2)*G*sin(state[0])*cos(del_) -
               (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
               (M1 + M2)*G*sin(state[2]))/den2

    return dydx


# create a time array at 0.001 second steps
dt = 0.001
t = np.arange(0.0, 250, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 135.0
w1  = 0.0
th2 = 135.0
w2  = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

y[:,0] = np.mod(y[:,0], 4*np.pi)
y[:,2] = np.mod(y[:,2], 4*np.pi)

U = -(M1+M2)*G*L1*np.cos(y[:,0])-M2*G*L2*np.cos(y[:,2])

# find minima and maxima
minima_idx = argrelextrema(U, np.less)[0]
maxima_idx = argrelextrema(U, np.greater)[0]

for idx in minima_idx:
    print(f"minima: theta1 = {y[idx,0]}, theta2 = {y[idx,2]}, U = {U[idx]}")

for idx in maxima_idx:
    print(f"maxima: theta1 = {y[idx,0]}, theta2 = {y[idx,2]}, U = {U[idx]}")


minima_data = [y[minima_idx,0],y[minima_idx,2],U[minima_idx]]
maxima_data = [y[maxima_idx,0],y[maxima_idx,2],U[maxima_idx]]

minima_data = np.delete(minima_data, -1, axis=1)

allpoints = np.concatenate([minima_data, maxima_data],axis=0)
saddle_idx = argrelextrema(allpoints[:,2], np.greater)[0]

grid1, grid2 = np.meshgrid(np.linspace(0, max(y[:,0]), 1000), np.linspace(0, max(y[:,2]), 1000))
gridU = griddata((y[:,0], y[:,2]), U, (grid1, grid2), method='cubic')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(grid1, grid2, gridU, color='yellow',alpha=0.5)
ax.scatter(y[minima_idx,0], y[minima_idx, 2], U[minima_idx], color = 'blue', label='minima')
ax.scatter(y[maxima_idx,0], y[maxima_idx, 2], U[maxima_idx], color = 'red', label='maxima')
ax.set_xlabel('theta1[rad]')
ax.set_ylabel('theta2[rad]')
ax.set_zlabel('potential[J]')
ax.set_title("theta1 -theta2 - potential")
ax.legend()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(grid1, grid2, gridU, color='red',alpha=1.0)
ax.set_xlabel('theta1[rad]')
ax.set_ylabel('theta2[rad]')
ax.set_zlabel('potential[J]')
plt.show()