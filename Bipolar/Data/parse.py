
import numpy as np
import pandas as pd

x = pd.read_csv("bipolar_rep.tsv", sep="\t").values

normalizer = np.max(np.abs(x))
x /= normalizer

np.savetxt("bipolar_rep_scaled.tsv", x, delimiter="\t")
