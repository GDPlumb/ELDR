
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()

X = data.data
X = X - np.min(X, axis = 0)
X = X / np.max(X, axis = 0)

y = data.target

np.savetxt("X.tsv", X, delimiter="\t")
np.savetxt("y.tsv", y, delimiter="\t")
