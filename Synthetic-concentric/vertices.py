
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

import json
import numpy as np
import pandas as pd
import pickle

num_clusters = 2

data_rep = pickle.load(open("Model/points.pkl", "rb" ))

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
