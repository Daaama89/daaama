import matplotlib.pyplot as plt
import numpy as np

# サンプルの時系列データ（仮のデータ）
time = np.arange(0, 10, 0.1)
y = np.sin(2 * np.pi * 1 * time)  # 仮のデータ
threshold = 0.5  # 閾値

# オーバーした部分を取得
highlight_indices = np.where(np.abs(y) > threshold)[0]

# プロット
plt.plot(time, y, label='Time Series Data')

# オーバーした部分を赤色でハイライト
for i in range(0, len(highlight_indices), 2):
    plt.axvspan(time[highlight_indices[i]], time[highlight_indices[i + 1]], color='red', alpha=0.3)

# プロットの設定
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Highlighting Above-threshold Regions in Time Series Data')

# 凡例の表示
plt.legend()

# グリッドの表示（オプション）
plt.grid(True)

# グラフの表示
plt.show()
def objective_func(coords):
    theta1, theta2 = coords
    dU_dtheta1 = (M1+M2)*G*L1*np.sin(theta1)
    dU_dtheta2 = -M2*G*L2*np.sin(theta2)
    return dU_dtheta1, dU_dtheta2

initial_guess = [y[-1,0], y[-1, 2]]

result = minimize(objective_func, initial_guess)
theta1_min = result.x[0]
theta2_max = result.x[0]
print(f"theta1 mini:{theta1_min}")
print(f"theta2 maxi:{theta2_max}")


# 時間と力学的エネルギーのプロット
plt.plot(t[:], (0.5*(M1+M2)*(L1**2)*((y[:,1])**2) + 0.5*M2*(L2**2)*((y[:,3])**2) + L1*L2*(y[:,1])*(y[:,3])*np.cos((y[:,0]) - (y[:,2])) - (M1+M2)*G*L1*np.cos(y_1[:,0])-M2*G*L2*np.cos(y_1[:,2])),'k',label = 'potential',linewidth = 1)
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('Mechanical Energy (J)',fontsize=18)
plt.xticks(fontsize=15)
plt.yticks([20.809, 20.81, 20.811], fontsize=15)
plt.show()

print("変更を加えたにょん")
print("ファイルチェックだにょん")


#doublependulum2.py

dU_dtheta1 = np.gradient(U, y[:,0])
dU_dtheta2 = np.gradient(U, y[:,2])
ddU_dtheta1 = np.gradient(dU_dtheta1, y[:,0])
ddU_dtheta2 = np.gradient(dU_dtheta2, y[:,2])

y_1 = np.zeros_like(y)

# 4pi
y_1[:,0] = np.mod(y[:,0] + 2*np.pi, 4*np.pi)
y_1[:,2] = np.mod(y[:,2] + 2*np.pi ,4*np.pi)

x1 = L1*sin(y_1[:, 0])
y1 = -L1*cos(y_1[:, 0])
x2 = L2*sin(y_1[:, 2]) + x1
y2 = -L2*cos(y_1[:, 2]) + y1

grid1, grid2 = np.meshgrid(np.linspace(0, 4*np.pi, 100),np.linspace(0, 4*np.pi, 100))
gridU = griddata((y_1[:,0], y_1[:,2]), U, (grid1, grid2), method='cubic')

# minimum(derivate)

minima_indices_all1 = np.where((np.diff(np.sign(dU_dtheta1))!=0) & (ddU_dtheta1[:-1]>0) & (U[:-1]<-9.81))[0]
minima_indices_all2 = np.where((np.diff(np.sign(dU_dtheta2))!=0) & (ddU_dtheta2[:-1]>0) & (U[:-1]<-9.81))[0]
minima_indices_all = np.union1d(minima_indices_all1, minima_indices_all2)

# minimum(minimize)
minima_indices = np.where(np.isclose(np.gradient(U,y[:,0]), 0, rtol=1e-3))[0]


#peak

changing_sign_indices1 = np.where(((np.diff(np.sign(dU_dtheta1))< 0) & (ddU_dtheta1[:-1]< 0) & (U[:-1]>9)) | ((np.diff(np.sign(dU_dtheta1))> 0) & (ddU_dtheta1[:-1]> 0) & (U[:-1]>9)))[0]
changing_sign_indices2 = np.where(((np.diff(np.sign(dU_dtheta2))< 0) & (ddU_dtheta2[:-1]< 0) & (U[:-1]>-10)) | ((np.diff(np.sign(dU_dtheta2))> 0) & (ddU_dtheta2[:-1]> 0) & (U[:-1]>-10)))[0]

common_indices = np.intersect1d(changing_sign_indices1, changing_sign_indices2)
non_common_c1 = np.setdiff1d(changing_sign_indices1, common_indices)
non_common_c2 = np.setdiff1d(changing_sign_indices2, common_indices)

changing_sign_indices = np.concatenate((non_common_c1, non_common_c2))
#plot potential

plt.figure()
plt.plot(t[:],-(M1+M2)*G*L1*np.cos(y_1[:,0])-M2*G*L2*np.cos(y_1[:,2]),'k',label = 'potential',linewidth = 1)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("potential (J)", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()
plt.plot(t[:],-(M1+M2)*G*L1*np.cos(y_1[:,0])-M2*G*L2*np.cos(y_1[:,2]),'k',label = 'potential',linewidth = 1)
plt.scatter(t[down_up], -(M1+M2)*G*L1*np.cos(y_1[down_up,0])-M2*G*L2*np.cos(y_1[down_up,2]), color='red', s=10)
plt.scatter(t[up_down], -(M1+M2)*G*L1*np.cos(y_1[up_down,0])-M2*G*L2*np.cos(y_1[up_down,2]), color='blue', s=10)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("potential (J)", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()
plt.plot(t[:50000],-(M1+M2)*G*L1*np.cos(y_1[:50000,0])-M2*G*L2*np.cos(y_1[:50000,2]),'k',label = 'potential',linewidth = 1)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("potential [(J)", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()
plt.plot(t[:],x1[:],'k',label = 'potential',linewidth = 1)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("x1", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()
plt.plot(t[:],x2[:],'k',label = 'potential',linewidth = 1)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("x2", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()
plt.plot(t[:],y1[:],'k',label = 'potential',linewidth = 1)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("y1", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()
plt.plot(t[:],y2[:],'k',label = 'potential',linewidth = 1)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("y2", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()
plt.scatter(y[:,0],y[:,2],color = 'black',label = 'theta1 - theta2',s=1)
plt.scatter(y[changing_sign_indices1,0], y[changing_sign_indices1,2], color = 'red', label = 'peak point_theta1')
plt.scatter(y[changing_sign_indices2,0], y[changing_sign_indices2,2], color = 'blue',label = 'peak point_theta2')
plt.xlabel("theta1(rad)",fontsize=18)
plt.ylabel("theta2(rad)",fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc = 'upper left')
plt.show()

plt.figure()
plt.scatter(y[:,0],y[:,2],color = 'black',label = 'theta1 - theta2',s=1)
plt.scatter(y[minima_indices_all1,0], y[minima_indices_all1,2], color = 'red', label = 'local minimun1')
plt.scatter(y[minima_indices_all2,0], y[minima_indices_all2,2], color = 'blue', label = 'local minimun2')
plt.xlabel("theta1(rad)",fontsize=18)
plt.ylabel("theta2(rad)",fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc = 'upper left')
plt.show()

plt.figure()
plt.scatter(y[:,0],y[:,2],color = 'black',label = 'theta1 - theta2',s=1)
plt.scatter(y[minima_indices_all,0], y[minima_indices_all,2], color = 'red', label = 'local minimum')
plt.xlabel("theta1(rad)",fontsize=18)
plt.ylabel("theta2(rad)",fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc = 'upper left')
plt.show()

# アニメーションの作成
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,8))
ax1.set_xlim(min(y[:,0]),max(y[:,0]))
ax1.set_ylim(min(U), max(U))
ax2.set_xlim(min(y[:,2]),max(y[:,2]))
ax2.set_ylim(min(U), max(U))

line1, =ax1.plot([],[],lw=2)
line2, =ax2.plot([],[],lw=2)

time_template = 'time = %.1fs'
time_text = ax1.text(0.05, 0.9,'',transform = ax1.transAxes)

def init():
    line1.set_data([],[])
    line2.set_data([],[])
    time_text.set_text('')
    return line1, line2, time_text

def animate(i):
    line1.set_data(y[:i,0], U[:i])
    line2.set_data(y[:i,2], U[:i])
    time_text.set_text(time_template % (i*dt))
    return line1, line2, time_text

ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
                              interval=dt*1000, blit=True, init_func=init)
plt.show()

#アニメーションの作成
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.set_xlim(min(y_1), max(y_1))
ax1.set_xlabel(r'$\theta_1[rad]$', fontsize=14)
ax1.tick_params(axis='x',labelsize=14)
ax1.set_ylim(min(U), max(U))
ax1.set_ylabel(r'$U[J]$', fontsize=14)
ax1.tick_params(axis='y',labelsize=14)

ax2.set_xlim(min(y_2), max(y_2))
ax2.set_xlabel(r'$\theta_2[rad]$', fontsize=14)
ax2.tick_params(axis='x',labelsize=14)
ax2.set_ylim(min(U), max(U))
ax2.set_ylabel(r'$U[J]$', fontsize=14)
ax2.tick_params(axis='y',labelsize=14)

line1, = ax1.plot([], [], lw=2)
line2, = ax2.plot([], [], lw=2)

time_template = 'time = %.1fs'
time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    time_text.set_text('')
    return line1, line2, time_text

# アニメーション時間を15秒までに制限
animation_time_limit = 15.0
num_frames = int(animation_time_limit / dt)

def animate(i):
    line1.set_data(y_1[:i], U[:i])
    line2.set_data(y_2[:i], U[:i])
    time_text.set_text(time_template % (i * dt))
    return line1, line2, time_text

ani = animation.FuncAnimation(fig, animate, range(1, num_frames),
                              interval=dt * 1000, blit=True, init_func=init)

ani.save('animated_pendulum.mp4', writer='ffmpeg', fps = int(1/dt))

plt.show()

# アニメーションの作成
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal', 'box')

line, = ax.plot([], [], 'o-', lw=2)

def update(frame):
    x = [0, L1 * np.sin(y[frame, 0]), L1 * np.sin(y[frame, 0]) + L2 * np.sin(y[frame, 2])]
    y_vals = [0, -L1 * np.cos(y[frame, 0]), -L1 * np.cos(y[frame, 0]) - L2 * np.cos(y[frame, 2])]
    line.set_data(x, y_vals)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(t), interval=dt * 1000)

# MP4形式で保存
ani.save('double_pendulum.mp4', writer='ffmpeg', fps = int(1/dt))
plt.show()

#0125
dU_dtheta1 = np.gradient(U, y[:,0])
dU_dtheta2 = np.gradient(U, y[:,2])
ddU_dtheta1 = np.gradient(dU_dtheta1, y[:,0])
ddU_dtheta2 = np.gradient(dU_dtheta2, y[:,2])


# minimum(derivate)

minima_indices_all1 = np.where((np.diff(np.sign(dU_dtheta1))!=0) & (ddU_dtheta1[:-1]>0) & (U[:-1]<-9.81))[0]
minima_indices_all2 = np.where((np.diff(np.sign(dU_dtheta2))!=0) & (ddU_dtheta2[:-1]>0) & (U[:-1]<-9.81))[0]
minima_indices_all = np.union1d(minima_indices_all1, minima_indices_all2)

# minimum(minimize)
minima_indices = np.where(np.isclose(np.gradient(U,y[:,0]), 0, rtol=1e-3))[0]
print(minima_indices)

#peak

changing_sign_indices1 = np.where(((np.diff(np.sign(dU_dtheta1))< 0) & (ddU_dtheta1[:-1]< 0) & (U[:-1]>=9.81)) | ((np.diff(np.sign(dU_dtheta1))> 0) & (ddU_dtheta1[:-1]> 0) & (U[:-1]>=9.81)))[0]
changing_sign_indices2 = np.where(((np.diff(np.sign(dU_dtheta2))< 0) & (ddU_dtheta2[:-1]< 0) & (U[:-1]>=-9.81)) | ((np.diff(np.sign(dU_dtheta2))> 0) & (ddU_dtheta2[:-1]> 0) & (U[:-1]>=-9.81)))[0]

common_indices = np.intersect1d(changing_sign_indices1, changing_sign_indices2)
non_common_c1 = np.setdiff1d(changing_sign_indices1, common_indices)
non_common_c2 = np.setdiff1d(changing_sign_indices2, common_indices)

changing_sign_indices = np.concatenate((non_common_c1, non_common_c2))



# saddle and non-eq
for i in range(-40,42):
  m_point_indices1 = np.where((U>=-9.81) & (y[:,0]<2*i*np.pi))[0]


grid1, grid2 = np.meshgrid(np.linspace(0, 4*np.pi, 100),np.linspace(0, 4*np.pi, 100))
gridU = griddata((y_1[:,0], y_1[:,2]), U, (grid1, grid2), method='cubic')

# 時間と力学的エネルギーのプロット
plt.plot(t[:], (0.5*(M1+M2)*(L1**2)*((y[:,1])**2) + 0.5*M2*(L2**2)*((y[:,3])**2) + L1*L2*(y[:,1])*(y[:,3])*np.cos((y[:,0]) - (y[:,2])) - (M1+M2)*G*L1*np.cos(y_1[:,0])-M2*G*L2*np.cos(y_1[:,2])),'k',label = 'potential',linewidth = 1)
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('Mechanical Energy (J)',fontsize=18)
plt.xticks(fontsize=15)
plt.yticks([20.809, 20.81, 20.811], fontsize=15)
plt.show()


#plot potential

plt.figure()
plt.plot(t[:],-(M1+M2)*G*L1*np.cos(y_1[:,0])-M2*G*L2*np.cos(y_1[:,2]),'k',label = 'potential',linewidth = 1)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("potential (J)", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()
plt.plot(t[:],-(M1+M2)*G*L1*np.cos(y_1[:,0])-M2*G*L2*np.cos(y_1[:,2]),'k',label = 'potential',linewidth = 1)
plt.scatter(t[minima_indices_all1], -(M1+M2)*G*L1*np.cos(y_1[minima_indices_all1,0])-M2*G*L2*np.cos(y_1[minima_indices_all1,2]), color='red', s=10)
plt.scatter(t[minima_indices_all2], -(M1+M2)*G*L1*np.cos(y_1[minima_indices_all2,0])-M2*G*L2*np.cos(y_1[minima_indices_all2,2]), color='blue', s=10)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("potential (J)", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()
plt.plot(t[:50000],-(M1+M2)*G*L1*np.cos(y_1[:50000,0])-M2*G*L2*np.cos(y_1[:50000,2]),'k',label = 'potential',linewidth = 1)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("potential [(J)", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()
plt.plot(t[:],x1[:],'k',label = 'potential',linewidth = 1)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("x1", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()
plt.plot(t[:],x2[:],'k',label = 'potential',linewidth = 1)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("x2", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()
plt.plot(t[:],y1[:],'k',label = 'potential',linewidth = 1)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("y1", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()
plt.plot(t[:],y2[:],'k',label = 'potential',linewidth = 1)
plt.xlabel("time (s)",fontsize=18)
plt.ylabel("y2", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


plt.figure()
plt.plot(y[:,0],U[:],'k',label = 'theta1 - potential',linewidth = 1)
plt.scatter(y[changing_sign_indices1,0], U[changing_sign_indices1], color = 'red',label = 'peak' ,s = 30)
plt.scatter(y[minima_indices_all1,0], U[minima_indices_all1], color = 'blue',label = 'minimum' ,s = 30)
plt.xlabel("theta1[rad]")
plt.ylabel("potential [J]")
plt.title("U - theta1")
plt.legend(loc = 'upper left')
plt.show()

plt.figure()
plt.plot(y[:,2],U[:],'k',label = 'theta1 - potential',linewidth = 1)
plt.scatter(y[changing_sign_indices2,2], U[changing_sign_indices2], color = 'red',label = 'min2' ,s = 30)
plt.scatter(y[minima_indices_all2,0], U[minima_indices_all2], color = 'blue',label = 'minimum' ,s = 30)
plt.xlabel("theta2[rad]")
plt.ylabel("potential [J]")
plt.title("U - theta2")
plt.legend(loc = 'upper left')
plt.show()

plt.figure()
plt.scatter(y[:,0],y[:,2],color = 'black',label = 'theta1 - theta2',s=1)
plt.scatter(y[changing_sign_indices1,0], y[changing_sign_indices1,2], color = 'red', label = 'peak point_theta1')
plt.scatter(y[changing_sign_indices2,0], y[changing_sign_indices2,2], color = 'blue',label = 'peak point_theta2')
plt.xlabel("theta1(rad)",fontsize=18)
plt.ylabel("theta2(rad)",fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc = 'upper left')
plt.show()

plt.figure()
plt.scatter(y[:,0],y[:,2],color = 'black',label = 'theta1 - theta2',s=1)
plt.scatter(y[minima_indices_all1,0], y[minima_indices_all1,2], color = 'red', label = 'local minimun1')
plt.scatter(y[minima_indices_all2,0], y[minima_indices_all2,2], color = 'blue', label = 'local minimun2')
plt.xlabel("theta1(rad)",fontsize=18)
plt.ylabel("theta2(rad)",fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc = 'upper left')
plt.show()

plt.figure()
plt.scatter(y[:,0],y[:,2],color = 'black',label = 'theta1 - theta2',s=1)
plt.scatter(y[minima_indices_all,0], y[minima_indices_all,2], color = 'red', label = 'local minimum')
plt.xlabel("theta1(rad)",fontsize=18)
plt.ylabel("theta2(rad)",fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc = 'upper left')
plt.show()

# data out
data =  L1*np.sin(y[:,0])
np.savetxt("x1.csv", data, delimiter=",", fmt ='%.18f')

data2 =  L1*np.sin(y[:,0])+L2*np.sin(y[:,2])
np.savetxt("x2.csv", data2, delimiter=",", fmt ='%.18f')

data3 = y_1[:,0]
np.savetxt("theta1.csv",data3,delimiter=",",fmt='%.18f')

data4 = y_1[:,2]
np.savetxt("theta2.csv",data4,delimiter=",",fmt='%.18f')

data5 = np.sin(y[:,2])
np.savetxt("sin_theta2.csv",data5,delimiter=",",fmt='%.18f')

data6 = np.sin(y[:,0])
np.savetxt("sin_theta1.csv",data6,delimiter=",",fmt='%.18f')

data7 = -L2*np.cos(y[:, 2]) - L1*np.cos(y[:,0])
np.savetxt("y2.csv",data7,delimiter=",",fmt='%.18f')

data8 = -L1*np.cos(y[:,0])
np.savetxt("y1.csv",data8,delimiter=",",fmt='%.18f')

data9 = U
np.savetxt("U.csv",data9,delimiter=",",fmt='%.18f')


data12 = y[:,1]
np.savetxt("omega1.csv",data12,delimiter=",",fmt='%.18f')

data13 = y[:,3]
np.savetxt("omega2.csv",data13,delimiter=",",fmt='%.18f')

data14 = dU_dtheta1
np.savetxt("dU_dtheta1.csv",data14,delimiter=",",fmt='%.18f')

data15 = dU_dtheta2
np.savetxt("dU_dtheta2.csv",data15,delimiter=",",fmt='%.18f')


data17 = changing_sign_indices2
np.savetxt("peak_theta2.csv", data17, delimiter=",", fmt='%d')

data18 = changing_sign_indices1
np.savetxt("peak_theta1.csv", data18, delimiter=",", fmt='%d')

data19 = changing_sign_indices
np.savetxt("all_peak_.csv", data19, delimiter=",", fmt='%d')

data20 = minima_indices_all2
np.savetxt("local_minimum_.csv", data20, delimiter=",", fmt='%d')

data21 = minima_indices_all
np.savetxt("local_minimum.csv", data21, delimiter=",", fmt='%d')











# data out
data =  L1*np.sin(y[:,0])
np.savetxt("x1.csv", data, delimiter=",", fmt ='%.18f')

data2 =  L1*np.sin(y[:,0])+L2*np.sin(y[:,2])
np.savetxt("x2.csv", data2, delimiter=",", fmt ='%.18f')

data3 = y_1[:,0]
np.savetxt("theta1.csv",data3,delimiter=",",fmt='%.18f')

data4 = y_1[:,2]
np.savetxt("theta2.csv",data4,delimiter=",",fmt='%.18f')

data5 = np.sin(y[:,2])
np.savetxt("sin_theta2.csv",data5,delimiter=",",fmt='%.18f')

data6 = np.sin(y[:,0])
np.savetxt("sin_theta1.csv",data6,delimiter=",",fmt='%.18f')

data7 = -L2*np.cos(y[:, 2]) - L1*np.cos(y[:,0])
np.savetxt("y2.csv",data7,delimiter=",",fmt='%.18f')

data8 = -L1*np.cos(y[:,0])
np.savetxt("y1.csv",data8,delimiter=",",fmt='%.18f')

data9 = U
np.savetxt("U.csv",data9,delimiter=",",fmt='%.18f')


data12 = y[:,1]
np.savetxt("omega1.csv",data12,delimiter=",",fmt='%.18f')

data13 = y[:,3]
np.savetxt("omega2.csv",data13,delimiter=",",fmt='%.18f')

data14 = dU_dtheta1
np.savetxt("dU_dtheta1.csv",data14,delimiter=",",fmt='%.18f')

data15 = dU_dtheta2
np.savetxt("dU_dtheta2.csv",data15,delimiter=",",fmt='%.18f')


data17 = changing_sign_indices2
np.savetxt("peak_theta2.csv", data17, delimiter=",", fmt='%d')

data18 = changing_sign_indices1
np.savetxt("peak_theta1.csv", data18, delimiter=",", fmt='%d')

data19 = changing_sign_indices
np.savetxt("all_peak_.csv", data19, delimiter=",", fmt='%d')

data20 = minima_indices_all2
np.savetxt("local_minimum_.csv", data20, delimiter=",", fmt='%d')

data21 = minima_indices_all
np.savetxt("local_minimum.csv", data21, delimiter=",", fmt='%d')