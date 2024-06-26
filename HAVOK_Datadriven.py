import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.signal import lsim,StateSpace


rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

beta = 8/3
sigma = 10
rho = 28
dt = 0.001
t = np.arange(0,500+dt,dt)

def lorenz(x_y_z,t0,sigma=sigma, beta=beta, rho=rho):
    x,y,z=x_y_z
    return [sigma*(y-x),x*(rho-z)-y,x*y-beta*z]

x0 = (0,1,20)
x_t = integrate.odeint(lorenz, x0, t,rtol=10**(-12),atol
=10**(-12)*np.ones_like(x0))
x, y, z = x_t.T
plt.plot(x, y, z,linewidth=1)

#show the figure
fig = plt.figure()
ax1 = fig.add_subplot(111,projection="3d")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.set_title("fig1.lorenz")
ax1.plot(x,y,z)

fig = plt.figure()
ax =fig.add_subplot(111)
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.plot(x)

stackmax = 100 
r = 15 
H = np.zeros((stackmax, np.size(x)-stackmax))

for k in range(stackmax):
     H[k,:] = x[k:-(stackmax-k)]
     
U,S,VT = np.linalg.svd(H,full_matrices=0)
V = VT.T
V1 = V


# use DMD
# make delay Hankel matrix
V1 = V1.T

Ur = U[:, :r]
Sr = np.diag(S[:r])
Vr = V1[:r, :]

plt.figure(figsize=(10, 6))
plt.plot(Vr[0,:])
plt.title("Time-delay Trajectories")
plt.xlabel("Time")
plt.ylabel("Vi")
plt.show()

V_1 = Vr[:,:-1].T
V_2 = Vr[:,1:].T
print("V1 shape",np.shape(V_1))
print("V2 shape",np.shape(V_2))

A_1 = np.linalg.lstsq(V_2, V_1, rcond=None)[0]
print(np.shape(A_1))

def create_difference_matrix(n,X):
    # make I
    identity_matrix = np.identity(n)

    # input A_1
    arbitrary_matrix = X

    # subtraction A_1 and I  
    difference_matrix = arbitrary_matrix - identity_matrix[:n, :n]

    return difference_matrix

result_matrix_A = create_difference_matrix(A_1.shape[0],A_1)
result_matrix_A = result_matrix_A / dt
A1 = result_matrix_A[:(r-1),:(r-1)]
B1 = result_matrix_A[:-1,(r-1)]

np.savetxt("HAVOK_lorenzA_dp.txt",A1,fmt='%.18f',delimiter=",")
np.savetxt("HAVOK_lorenzB_dp.txt",B1,fmt='%.18f',delimiter=",")

print("result A shape",np.shape(result_matrix_A))
print("result B shape",np.shape(B1))
# カラーマップを設定
colors = [(0, 'brown'),(0.35, 'yellow'),(0.5, 'white'), (1, 'blue')]
cmap = LinearSegmentedColormap.from_list('custom', colors)
x_ticks = np.arange(-0.5,A1.shape[1],1)
y_ticks = np.arange(-0.5,A1.shape[0],1)

# 行列のプロット
plt.subplot(1, 2, 1)
plt.imshow(A1, cmap=cmap, vmin=A1.min(), vmax=A1.max())
cbar = plt.colorbar()
cbar.set_label("Value")
plt.title("Matrix A")
plt.grid(True, color='black',linewidth=0.5)
plt.xticks(x_ticks,[])
plt.yticks(y_ticks,[])

# ベクトルのプロット
plt.subplot(1, 2, 2)
plt.grid()
plt.imshow(B1.reshape(-1,1), cmap=cmap, vmin=B1.min(), vmax=B1.max())
cbar = plt.colorbar()
cbar.set_label("Value")
plt.title("Vector B")
plt.tight_layout()
plt.grid(True, color='black',linewidth=0.5)
plt.xticks([])
plt.yticks(y_ticks,[])
plt.show()

Vr = Vr.T

r = 15
B1 = B1.T
B1 = B1.reshape(-1,1)

# システムの状態空間表現
sysNew1 = StateSpace(A1, B1, np.eye(r-1), np.zeros((r-1, 1)))
tspan1, y1, _ = lsim(sysNew1, Vr[:,(r-1)], np.arange(0, len(Vr)*dt, dt) ,Vr[0,:(r-1)])

plt.plot(t[:-stackmax],Vr[:, 0], 'k', label = 'original',linewidth = 2)
plt.plot(t[:-stackmax],y1[:,0], 'r', label = 'simu', linewidth = 2)
plt.xlabel("time")
plt.ylabel("v1")
plt.legend()
plt.show()

plt.plot(t[:50000],Vr[:50000, 0], 'k', label = 'original',linewidth = 2)
plt.plot(t[:50000],y1[:50000,0], 'r', label = 'simu', linewidth = 2)
plt.xlabel("time")
plt.ylabel("v1")
plt.legend()
plt.show()

plt.plot(t[220000:240000],Vr[220000:240000,0],'k',label = 'original',linewidth=2)
plt.xlabel("time")
plt.ylabel("v1")
plt.legend()
plt.show()

plt.plot(t[220000:240000],Vr[220000:240000,10],'r',label = 'forcing',linewidth=2)
plt.xlabel("time")
plt.ylabel("v11")
plt.legend()
plt.show()
