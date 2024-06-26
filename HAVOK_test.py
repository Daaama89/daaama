#1.3.1 use central difference
def fourth_order_central_difference(matrix, h):
    rows, cols = matrix.shape
    derivative_matrix = np.zeros((rows, cols))

    for i in range(cols):
        for j in range(rows):
            if j < 2:
                derivative_matrix[j, i] = (-matrix[j + 3, i] + 8 * matrix[j + 2, i] - 8 * matrix[j + 1, i] + matrix[j , i]) / (12 * h)
            elif j >= rows - 2:
                derivative_matrix[j, i] = (-matrix[j, i] + 8 * matrix[j - 1, i] - 8 * matrix[j - 2, i] + matrix[j - 3, i]) / (12 * h)
            else:
                derivative_matrix[j, i] = (matrix[j - 2, i] - 8 * matrix[j - 1, i] + 8 * matrix[j + 1, i]- matrix[j + 2, i]) / (12 * h)

    return derivative_matrix

dV_1 = fourth_order_central_difference(V,dt)

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

#figure central difference
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