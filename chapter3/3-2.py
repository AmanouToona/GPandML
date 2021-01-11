"""
ガウス過程と機械学習
chapter3 3.2 p59~
ガウス過程
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ====================================
x = np.arange(1, 10, 0.05)
mean = np.zeros(len(x))
sample = 5
# ====================================

# カーネル計算
def k(x1, x2):
    return np.exp(- (x1 - x2) ** 2)


K = np.zeros((len(x), len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        K[i, j] = k(x[i], x[j])


# サンプリング
data = np.random.multivariate_normal(mean, K, sample)


# 作図 --------------------------
# サンプル
fig = plt.figure()
ax1 = fig.add_subplot(121)
for i in range(sample):
    ax1.scatter(x, data[i], s=4)

# 共分散
ax2 = fig.add_subplot(122)
ax2.imshow(K, cmap="Blues")

plt.show()
