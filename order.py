import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 1次元時系列データのサンプル
data1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

# Hankel行列の行数（適切な値を選択）
num_rows = 5

data2 = data1[:-1]

# Hankel行列の構築
Hankel_matrix = np.zeros((num_rows, len(data2) - num_rows + 1))

for i in range(num_rows):
    Hankel_matrix[i, :] = data2[i:i + len(data2) - num_rows + 1]

print("Hankel",Hankel_matrix)
# 特異値分解 (SVD) を実行
U, S, Vt = np.linalg.svd(Hankel_matrix, full_matrices=False)

print("U",U)
print("S",S)
print("V",Vt)

# DMDのダイナミクスモードを抽出
r = 2  # 抽出するモードの数
Ur = U[:, :r]
Sr = np.diag(S[:r])
Vr = Vt[:r, :]

print("Ur",Ur)
print("Sr",Sr)
print("Vr",Vr)

# Hankel_matrix[:, 1:] の行数を Ur.T の列数に合わせる
data = data1[1:]
print("data=",data)

Hankel_next = np.zeros((num_rows, len(data) - num_rows +1))
for i in range(num_rows):
    Hankel_next[i, :] = data[i:i + len(data) - num_rows + 1]

print("Hankel",Hankel_matrix)

print("Hankel next",Hankel_next)

# DMD行列 A の計算
A = Hankel_next @ Vr.T @ np.linalg.inv(Sr) @ Ur.T

print("A",A)

def create_difference_matrix(n):
    # 単位行列を作成
    identity_matrix = np.identity(n)

    # 任意の行列を作成（ここでは単純に0で初期化）
    arbitrary_matrix = np.zeros((n, n))

    # 単位行列との差を取る
    difference_matrix = arbitrary_matrix - identity_matrix[:n, :n]

    return difference_matrix

# 例として、3x4の行列に対して単位行列との差を取る
result_matrix = create_difference_matrix(A.shape[0])
print(result_matrix)


















#1.use central difference
def central_difference(matrix, h):
    rows, cols = matrix.shape
    derivative_matrix = np.zeros((rows, cols))

    for i in range(cols):
        for j in range(rows):
            if j == 0:
                derivative_matrix[j, i] = (matrix[j + 1, i] - matrix[j, i]) / h
            elif j == rows - 1:
                derivative_matrix[j, i] = (matrix[j, i] - matrix[j - 1, i]) / h
            else:
                derivative_matrix[j, i] = (matrix[j + 1, i] - matrix[j - 1, i]) / (2*h)

    return derivative_matrix

dV_1 = central_difference(V,dt)

print("dV's shape",np.shape(dV_1))

Xi_1 = np.linalg.lstsq(V,dV_1,rcond=None)[0]
A_dp = Xi_1[:(r-1),:(r-1)].T
B_dp = Xi_1[-1,:(r-1)].T

A_dp = np.array(A_dp)
B_dp = np.array(B_dp)
print("A's shape",np.shape(A_dp))
print("B's shape",np.shape(B_dp))
np.savetxt("HAVOK_A_dp.txt",A_dp,fmt='%.18f',delimiter=",")
np.savetxt("HAVOK_B_dp.txt",B_dp,fmt='%.18f',delimiter=",")

# color map
colors = [(0, 'blue'),(0.5, 'white'),(1, 'brown')]
cmap = LinearSegmentedColormap.from_list('custom', colors)

x_ticks = np.arange(-0.5,A_dp.shape[1],1)
y_ticks = np.arange(-0.5,A_dp.shape[0],1)

# matrix plot
plt.subplot(1, 2, 1)
plt.imshow(A_dp, cmap=cmap, vmin=A_dp.min(), vmax=A_dp.max())
cbar = plt.colorbar()
cbar.set_label("Value")
plt.title("Matrix A")
plt.grid(True, color='black',linewidth=0.5)
plt.xticks(x_ticks,[])
plt.yticks(y_ticks,[])


# vector plot
plt.subplot(1, 2, 2)
plt.grid()
plt.imshow(B_dp.reshape(-1,1), cmap=cmap, vmin=B_dp.min(), vmax=B_dp.max())
cbar = plt.colorbar()
cbar.set_label("Value")
plt.title("Vector B")
plt.tight_layout()
plt.grid(True, color='black',linewidth=0.5)
plt.xticks([])
plt.yticks(y_ticks,[])

plt.show()



# use 4th order central difference

def central_difference_4th_order(f, x, h):
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h))/(12*h)

V_derivative = np.zeros_like(V)
for i in range(V.shape[0]):
    V_derivative[i,:] = central_difference_4th_order(lambda x,i=i:V[i, x.astype(int)], np.arange(V.shape[1]),dt)

print("V_derivative shape",np.shape(V_derivative))

Xi_2 = np.linalg.lstsq(V,V_derivative,rcond=None)[0]
A1_dp = Xi_2[:(r-1),:(r-1)].T
B1_dp = Xi_2[-1,:(r-1)].T

A1_dp = np.array(A1_dp)
B1_dp = np.array(B1_dp)
print("A's shape",np.shape(A1_dp))
print("B's shape",np.shape(B1_dp))
np.savetxt("HAVOK_A_dp.txt",A1_dp,fmt='%.18f',delimiter=",")
np.savetxt("HAVOK_B_dp.txt",B1_dp,fmt='%.18f',delimiter=",")


# color map
colors = [(0, 'blue'),(0.5, 'white'),(1, 'brown')]
cmap = LinearSegmentedColormap.from_list('custom', colors)

x_ticks = np.arange(-0.5,A1_dp.shape[1],1)
y_ticks = np.arange(-0.5,A1_dp.shape[0],1)

# matrix plot
plt.subplot(1, 2, 1)
plt.imshow(A1_dp, cmap=cmap, vmin=A1_dp.min(), vmax=A1_dp.max())
cbar = plt.colorbar()
cbar.set_label("Value")
plt.title("Matrix A")
plt.grid(True, color='black',linewidth=0.5)
plt.xticks(x_ticks,[])
plt.yticks(y_ticks,[])


# vector plot
plt.subplot(1, 2, 2)
plt.grid()
plt.imshow(B1_dp.reshape(-1,1), cmap=cmap, vmin=B1_dp.min(), vmax=B1_dp.max())
cbar = plt.colorbar()
cbar.set_label("Value")
plt.title("Vector B")
plt.tight_layout()
plt.grid(True, color='black',linewidth=0.5)
plt.xticks([])
plt.yticks(y_ticks,[])

plt.show()




v_0 = np.zeros(A2_dp.shape[0])


# def HAVOK v model
def linear_system(v):
    dvdt = np.dot(A2_dp, V1[:-1]) + B2_dp * V[-1]
    return dvdt

# initial situation
for i in range(r-1):
  v_0 =  V[i,0] 

# time span
t_span = np.arange(0.0, 30, dt)

# calclate
sol = odeint(linear_system, v_0, t_span, tfirst=True, args=())

# plot
plt.plot(t_span, sol)
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.legend([f'v{i}' for i in range(r-1)] + ['v_r'])
plt.show()


csv_file = open("x1.csv","r",encoding="ms932")
f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator='\r\n', quotechar='"', skipinitialspace=True)
list_of_time = list(f)
csv_file.close()

data = np.array(list_of_time)