
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

import sys
sys.path.insert(0, "../scvis/lib/scvis/")
from vae import GaussianVAE

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
    
    delta_global = tf.Variable(tf.zeros(shape = [1, input_dim]), name = "delta_global")
    delta_ind = tf.Variable(tf.zeros(shape = [num_points, input_dim]), name = "delta_ind")
    input = X + delta_global + delta_ind

    # Perform any feature transformation specified
    if feature_transform is not None:
        matrix =  np.float32(pd.read_csv(feature_transform, sep="\t", header = None).values)
        input = tf.matmul(input, matrix)

    # Compute the representation of our input
    vae = GaussianVAE(input, 1, architecture["inference"]["layer_size"], architecture["latent_dimension"], decoder_layer_size=architecture["model"]["layer_size"])
    rep, _ = vae.encoder(prob = 1.0)

    # Setup and restore the tf session
    sess = tf.Session()

    sess.run(delta_global.initializer)
    sess.run(delta_ind.initializer)

    old_vars = tf.trainable_variables()[2:]
    saver = tf.train.Saver(var_list = old_vars)
    saver.restore(sess, model_file)

    return sess, rep, X, delta_global, delta_ind

# https://gist.github.com/psycharo/ca7633f50b0aef8de0096a61d868fbf0
def pdist(X):
  """
  Computes pairwise distance between each pair of points
  Args:
    X - [N,D] matrix representing N D-dimensional vectors
  Returns:
    [N,N] matrix of (squared) Euclidean distances
  """
  x2 = tf.reduce_sum(X * X, 1, True)
  return x2 - 2 * tf.matmul(X, tf.transpose(X)) + tf.transpose(x2)

# https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
def pairwise_l2_norm2(x, y, scope=None):
    with tf.name_scope("pairwise_l2_norm2"):
        size_x = tf.shape(x)[0]
        size_y = tf.shape(y)[0]
        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, size_y]))

        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, size_x]))
        yy = tf.transpose(yy, perm=[2, 1, 0])

        diff = xx - yy
        square_diff = tf.square(diff)

        square_dist = tf.reduce_sum(square_diff, 1)

        return square_dist

def explain(load_model, x, y, indices, c1, c2,
            num_points = 100, dispersion = 2.0, lambda_global = 0.5, lambda_ind = 4.0,
            learning_rate = 0.001, min_epochs = 500, stopping_epochs = 200, tol = 0.01):
    
    name = str(c1) + "to" + str(c2)
    
    # Visualize the data
    fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(2,1,1)
    plt.scatter(y[:, 0], y[:, 1], s = 12)
    
    # Sample num_points in clusters c1 and c2
    indices_c1 = np.random.choice(indices[c1], num_points, replace = False)
    indices_c2 = np.random.choice(indices[c2], num_points, replace = False)

    points_c1 = x[indices_c1]
    
    targets_c2 = y[indices_c2]

    # Define the objective function
    sess, rep, X, delta_global, delta_ind = load_model(num_points)

    T = tf.placeholder(tf.float32, shape=[None, 2])
    l_t = tf.reduce_mean(tf.reduce_min(pairwise_l2_norm2(rep, T), axis = 1))
    tf.summary.scalar("loss/target", l_t)
    
    l_d = dispersion * tf.reduce_mean(1 / (1 + pdist(rep)))
    tf.summary.scalar("loss/dispersion", l_d)
    
    l_g = lambda_global * tf.reduce_mean(tf.abs(delta_global))
    tf.summary.scalar("loss/global", l_g)
    
    l_i = lambda_ind * tf.reduce_mean(tf.abs(delta_ind))
    tf.summary.scalar("loss/individual", l_i)

    loss_op = l_t + l_d + l_g + l_i
    tf.summary.scalar("loss/total", loss_op)
    
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter("tb/" + name, sess.graph)
    
    # Define the optimization process
    grad = tf.gradients(loss_op, [delta_global, delta_ind])
    new_global = delta_global.assign(delta_global - learning_rate * grad[0])
    new_ind = delta_ind.assign(delta_ind - learning_rate * grad[1])
    
    # Plot the chosen points before perturbing them
    y_c1 = sess.run(rep, feed_dict={X: points_c1})
    plt.scatter(y_c1[:,0], y_c1[:,1], marker = "v", c = "green", s = 64)

    # Find the explanation
    epoch = 0
    best_epoch = 0
    best_loss = np.inf
    saver = tf.train.Saver(max_to_keep = 1)
    while True:
    
        if epoch > min_epochs and epoch - best_epoch > stopping_epochs:
            break

        sess.run([new_global, new_ind], feed_dict={X: points_c1, T: targets_c2})
        
        if epoch % 10 == 0:
            summary, loss = sess.run([summary_op, loss_op], feed_dict={X: points_c1, T: targets_c2})
            writer.add_summary(summary, epoch)
        
            if loss < best_loss - tol:
                best_loss = loss
                best_epoch = epoch
                saver.save(sess, "./model.cpkt")

        epoch += 1

    writer.flush()

    # Restore to best model
    saver.restore(sess, "./model.cpkt")

    # Plot the chosen points after perturbing them
    y_c1 = sess.run(rep, feed_dict={X: points_c1})
    plt.scatter(y_c1[:,0], y_c1[:,1], marker = "v", c = "red", s = 64)

    # Plot the explanation (perturbation)
    plt.subplot(2,1,2)

    d_g, d_i = sess.run([delta_global, delta_ind])
    axis = np.array(range(x.shape[1]))

    plt.scatter(axis, d_g, label = "Delta Global", marker = "x")
    plt.scatter(axis, np.mean(d_i, axis = 0), label = "Delta Individual - Mean", marker = "1")
    plt.scatter(axis, np.var(d_i, axis = 0), label = "Delta Individual - Var", marker = "2")

    plt.legend(fancybox=True, framealpha = 0.5)

    plt.show()
    
    plt.close()
    
    # Return the global and individual explantions
    return np.ndarray.flatten(d_g), np.ndarray.flatten(d_i)

def apply(load_model, x, y, indices, c1, d_g, num_points = 50):

    # Visualize the data
    fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(2,1,1)
    plt.scatter(y[:, 0], y[:, 1], s = 12)
    
    # Sample num_points in cluster c1
    indices_c1 = np.random.choice(indices[c1], num_points, replace = False)

    points_c1 = x[indices_c1]
    
    # Load the model
    sess, rep, X, delta_global, delta_ind = load_model(num_points)
    
    # Plot the chosen points before perturbing them
    y_c1 = sess.run(rep, feed_dict={X: points_c1})
    plt.scatter(y_c1[:,0], y_c1[:,1], marker = "v", c = "green", s = 64)
    
    # Set the global explanation
    sess.run(delta_global.assign(d_g))

    # Plot the chosen points after perturbing them
    y_c1 = sess.run(rep, feed_dict={X: points_c1})
    plt.scatter(y_c1[:,0], y_c1[:,1], marker = "v", c = "red", s = 64)

    plt.show()
    
    plt.close()
