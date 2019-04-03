
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

###
# CONFIG
###

batch_size = 20

###
# Data Setup
###

def two2nine(input):
    out = []
    for i in range(input.shape[0]):
        x = input[i, 0]
        y = input[i, 1]
        out.append(np.array([x+y, x-y, x*y, x*x, y*y, x*x*y, x*y*y, x*x*x, y*y*y]))
    return np.array(out)

centers = two2nine(np.array([[-3,0],[2,0], [1.5,0], [2.5,0], [2,0.5], [2,-0.5]]))

centers_out = np.array([[-1.7427815,   2.5251963 ],
                         [ 3.8409333,  -3.2238045 ],
                         [ 2.0468214,  -0.74286616],
                         [ 6.0350876,  -6.158917  ],
                         [ 6.525142,   -0.09381351],
                         [ 0.3316005,  -5.3302374 ]]) #the learned representation of the cluster centers

normalizer = np.max(np.abs(two2nine(pd.read_csv("../scvis-dev/data/synthetic_2d_2200.tsv", sep="\t").values)), axis = 0)
normalizer = np.reshape(normalizer, (1, 9))

x = pd.read_csv("../scvis-dev/data/synthetic_9d_2200.tsv", sep="\t").values * normalizer

###
# Load the pretrained VAE and add our delta's
###

def load_vae(use_delta = False):
    tf.reset_default_graph()
    
    try:
        config_file_yaml = open("../scvis-dev/lib/scvis/config/model_config.yaml", "r")
        config = yaml.load(config_file_yaml)
        config_file_yaml.close()
    except yaml.YAMLError as exc:
        print("Error in the configuration file: {}".format(exc))

    architecture = config["architecture"]
    architecture.update({"input_dimension": 9})

    X = tf.placeholder(tf.float32, shape=[None, 9])
    if use_delta:
        delta_global = tf.Variable(tf.zeros(shape = [1, 9]), name = "delta_global")
        delta_ind = tf.Variable(tf.zeros(shape = [batch_size, 9]), name = "delta_ind")
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
        saver.restore(sess, "model/model/perplexity_10_regularizer_0.001_batch_size_512_learning_rate_0.01_latent_dimension_2_activation_ELU_seed_1_iter_3000.ckpt")
    else:
        saver = tf.train.Saver()
        saver.restore(sess, "model/model/perplexity_10_regularizer_0.001_batch_size_512_learning_rate_0.01_latent_dimension_2_activation_ELU_seed_1_iter_3000.ckpt")

    if use_delta:
        return sess, rep, X, delta_global, delta_ind
    else:
        return sess, rep, X

#sess, rep, X = load_vae(use_delta = False)
#z_mu = sess.run(rep, feed_dict={X: centers})
#print(z_mu)

def explain(points, target, dispersion = None, lambda_global = None, lambda_ind = None, name = "out.pdf"):
    # Visualize the clusters and find their centers
    sess, rep, X = load_vae(use_delta = False)
    
    plt.subplot(2,1,1)

    z_mu = sess.run(rep, feed_dict={X: x})
    plt.scatter(z_mu[:, 0], z_mu[:, 1], s = 8)

    z_mu = sess.run(rep, feed_dict={X: centers})
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c = ["black", "cyan", "red", "orange", "yellow", "purple"], marker = "x")

    # Define the objective function
    sess, rep, X, delta_global, delta_ind = load_vae(use_delta = True)

    T = tf.placeholder(tf.float32, shape=[None, 2])
    l_t = tf.losses.mean_squared_error(T, rep)
    tf.summary.scalar("loss/t", l_t)
    l_d = dispersion * tf.reduce_mean(1 / (1 + pdist(rep)))
    tf.summary.scalar("loss/d", l_d)
    l_g = lambda_global * tf.reduce_mean(tf.abs(delta_global / normalizer))
    tf.summary.scalar("loss/g", l_g)
    l_i = lambda_ind * tf.reduce_mean(tf.abs(delta_ind / normalizer))
    tf.summary.scalar("loss/i", l_i)

    loss = l_t + l_d + l_g + l_i
    grad = tf.gradients(loss, [delta_global, delta_ind])
    new_global = delta_global.assign(delta_global - 0.05 * grad[0])
    new_ind = delta_ind.assign(delta_ind - 0.05 * grad[1])
    
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter("tb/" + name, sess.graph)

    # Find the explanation
    points_rep = sess.run(rep, feed_dict={X: points})
    plt.scatter(points_rep[:,0], points_rep[:,1], marker = "v", c = "violet", s = 16)

    for i in range(500): #TODO:  stopping condition
        grad = sess.run([new_global, new_ind], feed_dict={X: points, T: target})
        if i % 10 == 0:
            summary = sess.run(summary_op, feed_dict={X: points, T: target})
            writer.add_summary(summary, i)

    points_rep = sess.run(rep, feed_dict={X: points})
    plt.scatter(points_rep[:,0], points_rep[:,1], marker = "^", c = "violet", s = 16)

    plt.subplot(2,1,2)

    d_g, d_i = sess.run([delta_global, delta_ind])
    d_g = d_g / normalizer
    d_i = d_i / normalizer
    axis = np.array(range(x.shape[1]))

    plt.scatter(axis, d_g, label = "Delta Global", marker = "x")
    plt.scatter(axis, np.mean(d_i, axis = 0), label = "Delta Individual - Mean", marker = "1")
    plt.scatter(axis, np.var(d_i, axis = 0), label = "Delta Individual - Var", marker = "2")

    plt.legend(fancybox=True, framealpha=0.5)

    plt.savefig(name + ".pdf")
    
    plt.close()
    
    return d_g, d_i

def map(c1, c2, dispersion = 1.5, lambda_global = 10.0, lambda_ind = 20.0, noise = 0.4):
    points = np.random.uniform(low = -1.0 * noise, high = noise, size = (batch_size, 9)) + centers[c1, :]
    target = np.reshape(centers_out[c2, :], (1, 2))
    return explain(points, target, dispersion = dispersion, lambda_global = lambda_global, lambda_ind = lambda_ind, name = str(c1) + "to" + str(c2))

global_0_1, ind = map(0,1, noise = 2.0)
global_1_0, _ = map(1,0, dispersion = 10.0)
global_1_2, _ = map(1,2)
global_2_1, _ = map(2,1)
global_0_2, _ = map(0,2, noise = 2.0)

sys.stdout = open("output.txt","wt")
print("From 0 to 1 and then 1 to 0")
print(np.round(global_0_1 + global_1_0, 3))
print("From 1 to 2 and then 2 to 1")
print(np.round(global_1_2 + global_2_1, 3))
print("From 0 to 1 and then 1 to 2 compared to from 0 to 2")
print(np.round(global_0_2 - (global_0_1 + global_1_2), 3))

map(2,4)
