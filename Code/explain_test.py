
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

import sys
sys.path.insert(0, "../scvis/lib/scvis/")
from vae import GaussianVAE

from explain import pairwise_l2_norm2, pdist

def load_vae(input_dim, model_file, num_points, feature_transform = None):

    tf.reset_default_graph()
    
    # Model Configuration
    try:
        config_file_yaml = open("../scvis/lib/scvis/config/model_config.yaml", "r")
        config = yaml.load(config_file_yaml, Loader = yaml.FullLoader)
        config_file_yaml.close()
    except yaml.YAMLError as exc:
        print("Error in the configuration file: {}".format(exc))

    architecture = config["architecture"]
    architecture.update({"input_dimension": input_dim})

    # Setup our modified input to the model
    X = tf.placeholder(tf.float32, shape=[None, input_dim])
    D = tf.placeholder(tf.float32, shape=[1, input_dim])
    
    input = X + D

    # Perform any feature transformation specified
    if feature_transform is not None:
        matrix =  np.float32(pd.read_csv(feature_transform, sep="\t", header = None).values)
        input = tf.matmul(input, matrix)

    # Compute the representation of our input
    vae = GaussianVAE(input, 1, architecture["inference"]["layer_size"], architecture["latent_dimension"], decoder_layer_size=architecture["model"]["layer_size"])
    rep, _ = vae.encoder(prob = 1.0)

    # Setup and restore the tf session
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    return sess, rep, X, D

def explain(load_model, x, y, indices,
            num_points = 100, dispersion = 2.0, lambda_global = 0.5,
            learning_rate = 0.001, min_epochs = 500, stopping_epochs = 200, tol = 0.01):
    
    n_input = x.shape[1]
    n_output = y.shape[1]
    num_clusters = len(indices)
    
    # Sample num_points from each cluster
    chosen = np.zeros((num_clusters, num_points), dtype = np.int32)
    for i in range(num_clusters):
        chosen[i, :] = np.random.choice(indices[i], num_points, replace = False)

    points = np.zeros((num_clusters, num_points, n_input))
    for i in range(num_clusters):
        for j in range(num_points):
            points[i, j] = x[chosen[i, j]]

    targets = np.zeros((num_clusters, num_points, n_output))
    for i in range(num_clusters):
        for j in range(num_points):
            targets[i, j] = y[chosen[i, j]]

    # Define the objective function
    sess, rep, X, D = load_model(num_points)

    T = tf.placeholder(tf.float32, shape=[None, n_output])
    l_t = tf.reduce_mean(tf.reduce_min(pairwise_l2_norm2(rep, T), axis = 1))
    tf.summary.scalar("loss/target", l_t)
    
    l_d = dispersion * tf.reduce_mean(1 / (1 + pdist(rep)))
    tf.summary.scalar("loss/dispersion", l_d)
    
    l_g = lambda_global * tf.reduce_mean(tf.abs(D))
    tf.summary.scalar("loss/global", l_g)

    loss_op = l_t + l_d + l_g
    tf.summary.scalar("loss/total", loss_op)
    
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter("tb/", sess.graph)
    
    # Define the optimization process
    grad = tf.gradients(loss_op, [D])

    # Find the explanation
    deltas = np.zeros((num_clusters - 1, n_input)) #Row i is the explanation for "Cluster 0 to Cluster i + 1"
    
    for iter in range(1001):

        # Choose the initial and target cluster
        initial, target = np.random.choice(num_clusters, 2, replace = False)

        p = points[initial, :, :]
        t = targets[target, :, :]
        
        if initial == 0:
            d = deltas[target - 1]
        elif target == 0:
            d = -1.0 * deltas[initial - 1]
        else:
            d = -1.0 * deltas[initial - 1] + deltas[target - 1]

        deltas_grad, summary = sess.run([grad, summary_op], feed_dict={X: p, T: t, D: np.reshape(d, (1, n_input))})
        writer.add_summary(summary, iter)

        deltas_grad = np.squeeze(deltas_grad[0])
        
        # Update the corresponding delta
        if initial == 0:
            deltas[target - 1] -= learning_rate * deltas_grad
        elif target == 0:
            deltas[initial - 1] += learning_rate * deltas_grad
        else:
            deltas[initial - 1] += learning_rate * 0.5 * deltas_grad
            deltas[target - 1] -= learning_rate * 0.5 * deltas_grad

    writer.flush()

    return deltas

def apply(load_model, x, y, indices, c1, d_g, num_points = 50):

    # Visualize the data
    fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(2,1,1)
    plt.scatter(y[:, 0], y[:, 1], s = 12)
    
    # Sample num_points in cluster c1
    indices_c1 = np.random.choice(indices[c1], num_points, replace = False)

    points_c1 = x[indices_c1]
    
    # Load the model
    sess, rep, X, D = load_model(num_points)
    d = np.zeros((1, x.shape[1]))
    
    # Plot the chosen points before perturbing them
    y_c1 = sess.run(rep, feed_dict={X: points_c1, D: d})
    plt.scatter(y_c1[:,0], y_c1[:,1], marker = "v", c = "green", s = 64)

    # Plot the chosen points after perturbing them
    y_c1 = sess.run(rep, feed_dict={X: points_c1, D: d_g})
    plt.scatter(y_c1[:,0], y_c1[:,1], marker = "v", c = "red", s = 64)

    plt.show()
    
    plt.close()
