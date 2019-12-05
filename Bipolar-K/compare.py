
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../Code/")

from misc import truncate

num_clusters = 18

K = [1000, 500, 250, 100, 50]

def load(deltas, k, initial, target):

    if initial == 0:
        d = deltas[target - 1]
    elif target == 0:
        d = -1.0 * deltas[initial - 1]
    else:
        d = -1.0 * deltas[initial - 1] + deltas[target - 1]
        
    d = truncate(d, k)
    
    return d

def compare(e_large, e_small):
    difference = 0
    for i in range(e_large.shape[0]):
        if e_small[i] != 0 and e_large[i] == 0:
            difference += np.abs(e_small[i])
    return difference / np.sum(np.abs(e_small))


for c in range(len(K) - 1):
    k1 = K[c]
    k2 = K[c + 1]
    
    d1 = np.load("save/deltas-" + str(k1) + ".npy")
    d2 = np.load("save/deltas-" + str(k2) + ".npy")

    out = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i != j:
                e1 = load(d1, k1, i, j)
                e2 = load(d2, k2, i, j)
                out[i, j] = compare(e1, e2)
                
    plt.imshow(out, interpolation = "none")
    plt.title("Percentage of " + str(k2) + " not in " + str(k1))
    plt.ylabel("Initial Group")
    plt.xlabel("Target Group")
    plt.colorbar()
    plt.savefig("compare/" + str(k1) + "-" + str(k2) + ".png")
    plt.close()
