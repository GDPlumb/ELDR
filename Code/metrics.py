
from matplotlib.path import Path
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import tensorflow as tf

def metrics(load_model, x, indices, deltas, epsilon):
    
    n_input = x.shape[1]
    num_clusters = len(indices)

    # Define the objective function
    sess, rep, X, D = load_model()

    correctness = np.zeros((num_clusters, num_clusters))
    coverage = np.zeros((num_clusters, num_clusters))
    for initial in range(num_clusters):
        for target in range(num_clusters):

                # Get the points in the initial cluster
                x_init = x[indices[initial]]
                
                # Construct the target region in the representation space
                x_target = x[indices[target]]
                
                # Construct the explanation between the initial and target regions
                if initial == target:
                    d = np.zeros((1, n_input))
                elif initial == 0:
                    d = deltas[target - 1]
                elif target == 0:
                    d = -1.0 * deltas[initial - 1]
                else:
                    d = -1.0 * deltas[initial - 1] + deltas[target - 1]
                
                # Find the representation of the initial points after they have been transformed
                rep_init = sess.run(rep, feed_dict={X: x_init, D: np.reshape(d, (1, n_input))})
                
                # Find the representation of the target points without any transformation
                rep_target = sess.run(rep, feed_dict={X: x_target, D: np.zeros((1, n_input))})
                
                # Calculate pairwise l2 distance
                dists = euclidean_distances(rep_init, Y = rep_target)
                
                # Find which pairs of points are within epsilon of each other
                close_enough = 1.0 * (dists <= epsilon)
                
                if initial == target:
                    # In this setting, every point is similar enough to itself
                    threshold = 2.0
                else:
                    threshold = 1.0

                correctness[initial, target] = np.mean(1.0 * (np.sum(close_enough, axis = 1) >= threshold))
                coverage[initial, target] = np.mean(1.0 * (np.sum(close_enough, axis = 0) >= threshold))

    return correctness, coverage
