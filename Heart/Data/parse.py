
import numpy as np
import pandas as pd

data = pd.read_csv("heart.csv")

y = data.target
X = data.drop("target", axis = 1)

X = X - np.min(X, axis = 0)
X = X / np.max(X, axis = 0)

np.savetxt("X.tsv", X, delimiter="\t")
np.savetxt("y.tsv", y, delimiter="\t")
