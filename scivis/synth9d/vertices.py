
import json
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from base import load_original

x = pd.read_csv("../scvis-dev/data/synthetic_9d_2200.tsv", sep="\t").values

sess, rep, X = load_original()

data_rep = sess.run(rep, feed_dict={X: x})

plt.figure(figsize=(20,10))
plt.scatter(data_rep[:, 0], data_rep[:, 1], s = 12)

all_vertices = []
for i in range(6):
    print("Please outline Cluster: " + str(i))
    vertices = plt.ginput(-1, show_clicks = True)
    all_vertices.append([list(v) for v in vertices])

plt.show()
plt.close()

with open("vertices.json", "w") as outfile:
    json.dump(all_vertices, outfile)
