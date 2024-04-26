import numpy as np

m = 100
X = 2 * np.random.rand(100, 1) 
X_b = np.c_[np.ones((100, 1)), X]
shuffled_indices = np.random.permutation(m)
print(X_b)
print(X_b[shuffled_indices])