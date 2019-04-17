
import numpy as np
import pandas as pd
 
 
x = pd.read_csv("../scvis-dev/data/bipolar_pca100.tsv", sep="\t").values

normalizer = np.max(np.abs(x))
x /= normalizer

np.savetxt("bipolar.tsv", x, delimiter="\t")
