
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import pickle
from sklearn.manifold import TSNE

x = pickle.load(open("data.pkl", "rb"))

tsne = TSNE()
tsne_results = tsne.fit_transform(x)

plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
plt.savefig("tsne.pdf")
plt.close()
