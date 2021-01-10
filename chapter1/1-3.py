import numpy as np

from numpy import random
from matplotlib import pyplot as plt


# サンプル数、　ノイズレベル============
n_sample = 20
noise_level = 0.1
# =====================================

# 信号の準備 ----------------------------------
x = np.arange(-4, 4, 8 / n_sample)

# 答え p32 より
y_origin = -0.065 + 0.068 * x + 0.022 * x ** 2 + 0.333 * np.sin(x) - 0.863 * np.cos(x)

# ノイズを加える
noise = random.normal(loc=0, scale=noise_level, size=n_sample)
y = y_origin + noise

# 近似式の導出 -------------------------------
# Phi の定義
def const(x):
    return 1


def power_one(x):
    return x


def power_two(x):
    return x ** 2


Phi = [const, power_one, power_two, np.sin, np.cos]

X = [[Phi[0](xx), Phi[1](xx), Phi[2](xx), Phi[3](xx), Phi[4](xx)] for xx in x]
# X = [[phi(xx) for phi in Phi] for xx in x]
X = np.array(X)

w = np.linalg.inv(X.T @ X) @ X.T @ y


# 作図 -------------------------
fig, ax = plt.subplots()

# サンプルの作図
ax.scatter(x, y, label='sample')

# 信号の作図
ax.plot(x, y_origin, label='origin')

# 近似式の作図
y_hat = X @ w
ax.plot(x, y_hat, linestyle='dashed', label='regression')

ax.legend()

plt.show()
