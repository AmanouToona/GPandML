import numpy as np
from numpy import random
from matplotlib import pyplot as plt

# =====================================
n_sample = 20
noise_level = 0.5
# =====================================

x = np.arange(n_sample) * 0.1

# 答え
y_origin = 1.77 * x + 0.42

# ノイズを付加する
noise = random.normal(loc=0, scale=noise_level, size=n_sample)
y = y_origin + noise

# 回帰の計算
xn = sum(x)
yn = sum(y)
xn2 = sum(x ** 2)
xyn = sum(x * y)

A = [[n_sample, xn], [xn, xn2]]
A_inf = np.linalg.inv(A), [yn, xyn]

ab = np.dot(np.linalg.inv(A), [yn, xyn])
a = ab[0]
b = ab[1]

# 作図
fig, ax = plt.subplots()
ax.plot(x, y_origin, label="origin")
ax.scatter(x, y, label="sample")
ax.plot(x, a + b * x, linestyle="dashed", label="regression")

ax.legend()
plt.show()

