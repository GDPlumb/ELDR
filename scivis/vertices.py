
import json
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load import load_original

def run(data_file, input_dim, model_file, num_clusters):

    x = pd.read_csv(data_file, sep="\t").values

    sess, rep, X = load_original(input_dim, model_file)

    data_rep = sess.run(rep, feed_dict = {X: x})

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
