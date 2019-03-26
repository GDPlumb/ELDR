
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

import sys
sys.path.insert(0, "../scvis-dev/lib/scvis/")
from model import SCVIS
from vae import GaussianVAE

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

# CONFIG
config_file = "../scvis-dev/lib/scvis/config/model_config.yaml"
input_dim = 2
batch_size = 20
model_file = "output/model/perplexity_10_regularizer_0.001_batch_size_512_learning_rate_0.01_latent_dimension_2_activation_ELU_seed_1_iter_3000.ckpt"
data_file = "../scvis-dev/data/synthetic_2d_2200.tsv"

# Input Data
x = pd.read_csv(data_file, sep="\t").values
normalizer = np.max(np.abs(x))

# Cluster Centers
centers = np.array([[-3,0],[2,0], [1.5,0], [2.5,0], [2,0.5], [2,-0.5]])
centers_out = np.array([[-1.3374573,  -0.8798897 ],
                         [ 2.1395469,  1.9266412 ],
                         [ 0.553436,    1.0889692 ],
                         [ 3.8631773,   2.8151867 ],
                         [ 0.62358487,  3.9877918 ],
                         [ 3.0373228,  -0.5326907 ]])

# Load their pretrained VAE (we don't need any of the auxilary stuff they introduced to train their model and they don't expose the input)
def load_vae(use_delta = False):
    tf.reset_default_graph()
    
    try:
        config_file_yaml = open(config_file, "r")
        config = yaml.load(config_file_yaml)
        config_file_yaml.close()
    except yaml.YAMLError as exc:
        print("Error in the configuration file: {}".format(exc))

    architecture = config["architecture"]
    architecture.update({"input_dimension": input_dim})

    X = tf.placeholder(tf.float32, shape=[None, input_dim])
    if use_delta:
        delta_global = tf.Variable(tf.zeros(shape = [1, input_dim]), name = "delta_global")
        delta_ind = tf.Variable(tf.zeros(shape = [batch_size, input_dim]), name = "detlta_ind")
        input = (X + delta_global + delta_ind) / normalizer
    else:
        input = X / normalizer

    vae = GaussianVAE(input, batch_size, architecture['inference']['layer_size'], architecture['latent_dimension'], decoder_layer_size=architecture['model']['layer_size'])
    rep, _ = vae.encoder(prob = 1.0)

    sess = tf.Session()

    if use_delta:
        sess.run(delta_global.initializer)
        sess.run(delta_ind.initializer)
        old_vars = tf.trainable_variables()[2:]
        saver = tf.train.Saver(var_list = old_vars)
        saver.restore(sess, model_file)
    else:
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

    if use_delta:
        return sess, rep, X, delta_global, delta_ind
    else:
        return sess, rep, X

def explain(points, target, dispersion = 0.75, lambda_global = 0.01, lambda_ind = 0.3, name = "out.png"):
    # Visualize the clusters and find their centers
    sess, rep, X = load_vae(use_delta = False)

    z_mu = sess.run(rep, feed_dict={X: x})
    plt.scatter(z_mu[:, 0], z_mu[:, 1], s = 8)

    z_mu = sess.run(rep, feed_dict={X: centers})
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c = ["black", "cyan", "red", "orange", "yellow", "purple"], marker = "x")

    # Define the objective function
    sess, rep, X, delta_global, delta_ind = load_vae(use_delta = True)

    T = tf.placeholder(tf.float32, shape=[None, 2])
    loss = ( tf.losses.mean_squared_error(T, rep) +
            dispersion * tf.reduce_mean(1 / (1 + pdist(rep))) +
            lambda_global * tf.reduce_mean(tf.abs(delta_global)) +
            lambda_ind * tf.reduce_mean(tf.abs(delta_ind)) )
    grad = tf.gradients(loss, [delta_global, delta_ind])
    new_global = delta_global.assign(delta_global - 0.05 * grad[0])
    new_ind = delta_ind.assign(delta_ind - 0.05 * grad[1])

    # Find the explanation
    points_rep = sess.run(rep, feed_dict={X: points})
    plt.scatter(points_rep[:,0], points_rep[:,1], marker = "v", c = "violet", s = 16)

    for i in range(500): #TODO:  stopping condition
        grad = sess.run([new_global, new_ind], feed_dict={X: points, T: target})

    points_rep = sess.run(rep, feed_dict={X: points})
    plt.scatter(points_rep[:,0], points_rep[:,1], marker = "^", c = "violet", s = 16)

    plt.savefig(name)
    
    plt.close()

    return sess.run(delta_global), sess.run(delta_ind)

def map(c1, c2, dispersion = 0.5, lambda_global = 0.01, lambda_ind = 0.3, noise = 0.2):
    points = np.random.uniform(low = -1.0 * noise, high = noise, size = (batch_size,2)) + centers[c1, :]
    target = np.reshape(centers_out[c2, :], (1,2))
    return explain(points, target, dispersion = dispersion, lambda_global = lambda_global, lambda_ind = lambda_ind, name = str(c1) + "to" + str(c2))

global_0_1, _ = map(0,1, noise = 1.0)
global_1_0, _ = map(1,0, dispersion = 5.0)
global_1_2, _ = map(1,2)
global_2_1, _ = map(2,1)
global_0_2, _ = map(0,2, noise = 1.0)

sys.stdout = open("output.txt","wt")
print("From 0 to 1 and then 1 to 0")
print(global_0_1 + global_1_0)
print("From 1 to 2 and then 2 to 1")
print(global_1_2 + global_2_1)
print("From 0 to 1 and then 1 to 2 compared to from 0 to 2")
print(global_0_2 - (global_0_1 + global_1_2))
