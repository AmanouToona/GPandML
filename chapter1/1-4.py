"""
ガウス過程と機械学習
chapter1 1.4  p32~
ridge 回帰
"""
import numpy as np
from matplotlib import pyplot as plt


# データ =======================
X = [[1, 2, 4], [1, 3, 6], [1, 4, 8]]
y = [1, 2, 3]
alpha = 0.1
# ==============================

X = np.array(X)
y = np.array(y)

# w の導出
w = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[0])) @ X.T @ y

print(w)

