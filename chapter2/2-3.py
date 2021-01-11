"""
ガウス過程と機械学習
chapter2 2.1 p39~
ガウス分布  サンプルの作図
"""
import numpy as np
import matplotlib.pyplot as plt

# ==========================
mean = np.array([0, 0])
cov = np.array([[1, 0], [0, 1]])
sample = 1000
# ==========================

data = np.random.multivariate_normal(mean, cov, sample)

# 作図
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1])
plt.show()

