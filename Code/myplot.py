
import matplotlib
matplotlib.rc("xtick", labelsize = 24)
matplotlib.rc("ytick", labelsize = 24)
matplotlib.rc("axes", titlesize = 48)
matplotlib.rc("axes", labelsize = 48)
matplotlib.rc("lines", markersize = 16)

from matplotlib.path import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

def plot_polys(data_rep, vertices):

    num_clusters = len(vertices)

    fig, ax = plt.subplots(figsize=(20, 10))
    patches = []

    for i in range(num_clusters):
        line = plt.Polygon(vertices[i], closed = False, color="blue", alpha=0.3)
        ax.add_line(line)

    plt.scatter(data_rep[:, 0], data_rep[:, 1])

    plt.show()
    plt.close()

def plot_groups(x, data_rep, num_clusters, labels, contour = None, name = "plot_groups.png"):

    n = x.shape[0]
    cluster = -1.0 * np.ones((n))
    
    indices = [[]] * num_clusters
    centers = [[]] * num_clusters
    means = [[]] * num_clusters
    for i in range(num_clusters):
        indices[i] = []
        for j in range(n):
            if labels[j] == i:
                cluster[j] = i
                indices[i].append(j)
        means[i] = np.mean(x[indices[i], :], axis = 0)
        centers[i] = np.mean(data_rep[indices[i], :], axis = 0)
        
    centers = np.array(centers)
    means = np.array(means)

    fig, ax = plt.subplots(figsize=(20, 10))
    
    patches = []
    
    plt.scatter(data_rep[:, 0], data_rep[:, 1], c = cluster, cmap = plt.cm.coolwarm)

    for i in range(num_clusters):
        plt.text(centers[i, 0], centers[i, 1], str(i), fontsize = 72)
        
    if contour is not None:
        feature_0 = contour[0]
        feature_1 = contour[1]
        map = contour[2]
        plt.contour(feature_0, feature_1, map)
        plt.colorbar()

    plt.savefig(name)
    plt.show()
    plt.close()

    return means, centers, indices
    
def plot_metrics(a, b, name = "plot_metrics.png"):

    # Set up figure and image grid
    fig = plt.figure(figsize=(20, 10))
        
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1,2),
                     axes_pad=0.75,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.25,
                     )

    # Add data to image grid
    c = 0
    for ax in grid:
        if c == 0:
            im = ax.imshow(a, cmap = "RdYlGn", interpolation = "none", vmin = 0.0, vmax = 1.0)
            ax.set_title("Correctness - " + str(np.round(np.mean(a), 3)))
            ax.set_ylabel("Initial Group")
        elif c == 1:
            im = ax.imshow(b, cmap = "RdYlGn", interpolation = "none", vmin = 0.0, vmax = 1.0)
            ax.set_title("Coverage - "  + str(np.round(np.mean(b), 3)))
        ax.set_xlabel("Target Group")
        c += 1

    # Colorbar
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)

    plt.savefig(name)
    plt.show()
    plt.close()

def plot_explanation(load_model, x, data_rep, indices, deltas, a, b, c1, c2,  num_points = 50, name = "plot_explanation.png"):

    # Find the explanation from c1 to c2
    if c1 == 0:
        d = deltas[c2 - 1]
    elif c2 == 0:
        d = -1.0 * deltas[c1 - 1]
    else:
        d = -1.0 * deltas[c1 - 1] + deltas[c2 - 1]
    d = np.reshape(d, (1, d.shape[0]))
   
    # Visualize the data
    fig, ax = plt.subplots(figsize=(20, 30))
    fig.subplots_adjust(hspace = .3)

    for i in range(2):
        if i == 0:
            initial = c1
            target = c2
            sign = 1.0
        elif i == 1:
            initial = c2
            target = c1
            sign = -1.0

        # Plot the full representation
        ax = plt.subplot(3, 1, i + 1)
        
        plt.scatter(data_rep[:, 0], data_rep[:, 1])
    
        # Sample num_points in initial group
        indices_initial = np.random.choice(indices[initial], num_points, replace = False)
        points_initial = x[indices_initial, :]
    
        # Load the model
        sess, rep, X, D = load_model()
        d_zeros = np.zeros(d.shape)
    
        # Plot the chosen points before perturbing them
        y_initial = sess.run(rep, feed_dict={X: points_initial, D: d_zeros})
        plt.scatter(y_initial[:,0], y_initial[:,1], marker = "v", c = "green")
    
        # Plot the chosen points after perturbing them
        y_after = sess.run(rep, feed_dict={X: points_initial, D: sign * d})
        plt.scatter(y_after[:,0], y_after[:,1], marker = "v", c = "red")
    
        plt.title("Mapping from Group " + str(initial) + " to Group " + str(target) + "\n Correctness - " + str(np.round(a[initial, target], 3)) + ", Coverage - " + str(np.round(b[initial, target], 3)))
    
    ax = plt.subplot(3, 1, 3)

    feature_index = np.array(range(d.shape[1]))
    plt.scatter(feature_index, d)
    plt.title("Explanation for Group " + str(c1) + " to Group " + str(c2))
    plt.ylabel("Change applied")
    plt.xlabel("Feature Index")

    plt.savefig(name)
    plt.show()
    plt.close()


def plot_similarity(deltas, deltas_original, name = "plot_similarity.png"):

    num_clusters = deltas_original.shape[0] + 1

    print(np.round(deltas_original, 2))
    print(np.round(deltas[:num_clusters - 1, ], 2))
    
    diff = np.abs(deltas_original - deltas[:num_clusters - 1, ])
    
    plt.figure(figsize=(20, 10))
    
    plt.ylabel("Group Index")
    plt.xlabel("Feature Index")
    
    plt.imshow(diff, vmin = 0.0, vmax = np.max(np.abs(deltas_original)))

    plt.colorbar()
    
    plt.savefig(name)
    plt.show()
    plt.close()
