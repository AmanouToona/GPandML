import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

# データペア　===================================
D = ([[1, 2], 4], [[-1, 1], 2], [[3, 0], 1], [[-2, -2], -1])
# ========================================

# データペアから X, y 行列作製
X = list()
y = list()
for d in D:
    X.append([1] + d[0])
    y.append(d[1])

X = np.array(X)
y = np.array(y)

# w の導出
w = np.linalg.inv(X.T @ X) @ X.T @ y

# 作図
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x1', size=14)
ax.set_ylabel('x2', size=14)
ax.set_zlabel('y', size=14)

# データ点作図
ax.plot(X[:, 1], X[:, 2], y, 'o', color='springgreen')

# 近似平面作図
ax1 = [i for i in range(-3, 4)]  # axis x1
ax2 = [i for i in range(-3, 4)]  # axis x2
axx1, axx2 = np.meshgrid(ax1, ax2)

axx = [[1] * len(ax1) ** 2, list(itertools.chain.from_iterable(axx1)), list(itertools.chain.from_iterable(axx2))]
axx = np.array(axx).T

ax.plot_surface(axx1, axx2, (axx @ w).reshape(len(ax1), len(ax2)), alpha=0.5)

plt.show()
