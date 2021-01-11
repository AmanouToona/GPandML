"""
ガウス過程と機械学習
chapter3 3.3 p71~
ガウス過程 各種カーネルのサンプルと共分散行列
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ToDo 2次元のxに拡張する

# ====================================
x = np.arange(-5, 6, 0.05)
mean = np.zeros(len(x))
sample = 3
# ====================================

# カーネル計算
def k_rbf(x1, x2):
    return np.exp(- (x1 - x2) ** 2)


def k_linear(x1, x2):
    return x1 * x2


def k_exponential(x1, x2):
    theta1 = 1
    return np.exp(- abs(x1 - x2) / theta1)


def k_periodic(x1, x2):
    theta1 = 1
    theta2 = 1
    return np.exp(theta1 * np.cos(abs(x1 - x2) / theta2))


K = np.zeros((len(x), len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        # K[i, j] = k_rbf(x[i], x[j])
        # K[i, j] = k_linear(x[i], x[j])
        # K[i, j] = k_exponential(x[i], x[j])
        K[i, j] = k_periodic(x[i], x[j])


# サンプリング
data = np.random.multivariate_normal(mean, K, sample)


# 作図 --------------------------
# サンプル
fig = plt.figure()
ax1 = fig.add_subplot(121)
for i in range(sample):
    ax1.plot(x, data[i])

# 共分散
ax2 = fig.add_subplot(122)
ax2.imshow(K, cmap="Blues")

plt.show()
