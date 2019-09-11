
from matplotlib.path import Path
import numpy as np
import tensorflow as tf

def eval_correctness(load_model, x, indices, deltas, all_vertices):
    
    n_input = x.shape[1]
    num_clusters = len(indices)

    # Define the objective function
    sess, rep, X, D = load_model()

    correctness = np.ones((num_clusters, num_clusters))
    for initial in range(num_clusters):
        for target in range(num_clusters):
            if initial != target:

                # Get the points in the initial cluster
                points = x[indices[initial]]
                count = points.shape[0]

                # Construct the target region in the representation space
                path = Path(all_vertices[target])
                
                # Construct the explanation between the initial and target regions
                if initial == 0:
                    d = deltas[target - 1]
                elif target == 0:
                    d = -1.0 * deltas[initial - 1]
                else:
                    d = -1.0 * deltas[initial - 1] + deltas[target - 1]
                
                # Find the representation of the initial points after they have been transformed
                projected = sess.run(rep, feed_dict={X: points, D: np.reshape(d, (1, n_input))})

                # Create a score
                successes = 0
                for i in range(count):
                    if path.contains_points(projected[i].reshape((1,2))):
                        successes += 1
                correctness[initial, target] = successes / count

    return correctness
