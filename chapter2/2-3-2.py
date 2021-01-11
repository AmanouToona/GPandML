"""
ガウス過程と機械学習
chapter2 2.1 p39~
ガウス分布  確率分布の作図
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ==========================
mu = np.array([0, 0])
cov = np.array([[1, 0], [0, 1]])

x1 = x2 = np.arange(-5, 6, 0.1)
x1, x2 = np.meshgrid(x1, x2)
# ==========================


def gaussian(x):
    D = cov.ndim
    det = np.linalg.det(cov)

    return (1 / np.sqrt(2 * np.pi) ** D * np.sqrt(det)) * np.exp(-0.5 * np.diag((x - mu) @ np.linalg.inv(cov) @ (x - mu).T))


X = np.stack([np.ravel(x1), np.ravel(x2)], axis=0).T
N = gaussian(X)
N = N.reshape(x1.shape)

# 作図
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x1', size=14)
ax.set_ylabel('x2', size=14)
ax.set_zlabel('y', size=14)

ax.plot_surface(x1, x2, N, cmap=mpl.cm.coolwarm)
plt.show()


