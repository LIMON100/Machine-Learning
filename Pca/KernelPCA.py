import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles

np.random.seed(0)

x , y = make_circles(n_samples = 400 , factor = 0.1 , noise = 0.05)

plt.figure()
plt.scatter(x[:,0] , x[:,1])


