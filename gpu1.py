import numpy as np

x = np.random.randn(10000000).astype(np.float32)
y = np.random.randn(10000000).astype(np.float32)

print(x+y)


import cupy as cp

x = np.random.randn(10000000).astype(np.float32)
y = np.random.randn(10000000).astype(np.float32)

print(x+y)