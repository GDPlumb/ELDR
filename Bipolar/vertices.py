
import json
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "../")
from load_vae import load_vae

data_file = "Data/bipolar_rep_scaled.tsv"
input_dim = 100
model_file = "Model/model/perplexity_10_regularizer_0.001_batch_size_512_learning_rate_0.01_latent_dimension_2_activation_ELU_seed_1_iter_5400.ckpt"
num_clusters = 18

x = pd.read_csv(data_file, sep="\t").values

sess, rep, X, D = load_vae(input_dim, model_file)

d = np.zeros((1, x.shape[1]))

data_rep = sess.run(rep, feed_dict = {X: x, D: d})

plt.figure(figsize=(20,10))
plt.scatter(data_rep[:, 0], data_rep[:, 1], s = 8)

all_vertices = []
for i in range(num_clusters):
    print("Please outline Cluster: " + str(i))
    vertices = plt.ginput(-1, show_clicks = True)
    all_vertices.append([list(v) for v in vertices])

plt.show()
plt.close()

with open("vertices.json", "w") as outfile:
    json.dump(all_vertices, outfile)
