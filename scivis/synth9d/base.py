
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

import sys
sys.path.insert(0, "../scvis-dev/lib/scvis/")
from vae import GaussianVAE

def two2nine(input):
    out = []
    for i in range(input.shape[0]):
        x = input[i, 0]
        y = input[i, 1]
        out.append(np.array([x+y, x-y, x*y, x*x, y*y, x*x*y, x*y*y, x*x*x, y*y*y]))
    return np.array(out)

def load_original():
    tf.reset_default_graph()
    
    try:
        config_file_yaml = open("../scvis-dev/lib/scvis/config/model_config.yaml", "r")
        config = yaml.load(config_file_yaml, Loader = yaml.FullLoader)
        config_file_yaml.close()
    except yaml.YAMLError as exc:
        print("Error in the configuration file: {}".format(exc))

    architecture = config["architecture"]
    architecture.update({"input_dimension": 9})

    X = tf.placeholder(tf.float32, shape=[None, 9])

    vae = GaussianVAE(X, 1, architecture['inference']['layer_size'], architecture['latent_dimension'], decoder_layer_size=architecture['model']['layer_size'])
    rep, _ = vae.encoder(prob = 1.0)

    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, "model/model/perplexity_10_regularizer_0.001_batch_size_512_learning_rate_0.01_latent_dimension_2_activation_ELU_seed_1_iter_3000.ckpt")

    return sess, rep, X

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

def load_vae(num_points):
    tf.reset_default_graph()
    
    try:
        config_file_yaml = open("../scvis-dev/lib/scvis/config/model_config.yaml", "r")
        config = yaml.load(config_file_yaml, Loader = yaml.FullLoader)
        config_file_yaml.close()
    except yaml.YAMLError as exc:
        print("Error in the configuration file: {}".format(exc))

    architecture = config["architecture"]
    architecture.update({"input_dimension": 9})

    X = tf.placeholder(tf.float32, shape=[None, 9])
    delta_global = tf.Variable(tf.zeros(shape = [1, 9]), name = "delta_global")
    delta_ind = tf.Variable(tf.zeros(shape = [num_points, 9]), name = "delta_ind")
    input = X + delta_global + delta_ind

    vae = GaussianVAE(input, 1, architecture['inference']['layer_size'], architecture['latent_dimension'], decoder_layer_size=architecture['model']['layer_size'])
    rep, _ = vae.encoder(prob = 1.0)

    sess = tf.Session()

    sess.run(delta_global.initializer)
    sess.run(delta_ind.initializer)
    old_vars = tf.trainable_variables()[2:]
    saver = tf.train.Saver(var_list = old_vars)
    saver.restore(sess, "model/model/perplexity_10_regularizer_0.001_batch_size_512_learning_rate_0.01_latent_dimension_2_activation_ELU_seed_1_iter_3000.ckpt")

    return sess, rep, X, delta_global, delta_ind

def explain(x, y, indices, c1, c2, num_points = 20, dispersion = 1.5, lambda_global = 0.5, lambda_ind = 4):
    name = str(c1) + "to" + str(c2)
    
    # Visualize the data
    fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(2,1,1)
    plt.scatter(y[:, 0], y[:, 1], s = 12)
    
    # Sample num_points in cluster c1 from x
    points = x[np.random.choice(indices[c1], num_points, replace = False)]
    
    # The target point is the center of cluster c2 in y
    target = np.mean(y[indices[c2], :], axis = 0).reshape((1,2))

    # Define the objective function
    sess, rep, X, delta_global, delta_ind = load_vae(num_points)

    T = tf.placeholder(tf.float32, shape=[None, 2])
    l_t = tf.losses.mean_squared_error(T, rep)
    tf.summary.scalar("loss/t", l_t)
    l_d = dispersion * tf.reduce_mean(1 / (1 + pdist(rep)))
    tf.summary.scalar("loss/d", l_d)
    l_g = lambda_global * tf.reduce_mean(tf.abs(delta_global))
    tf.summary.scalar("loss/g", l_g)
    l_i = lambda_ind * tf.reduce_mean(tf.abs(delta_ind))
    tf.summary.scalar("loss/i", l_i)

    loss = l_t + l_d + l_g + l_i
    grad = tf.gradients(loss, [delta_global, delta_ind])
    new_global = delta_global.assign(delta_global - 0.05 * grad[0])
    new_ind = delta_ind.assign(delta_ind - 0.05 * grad[1])
    
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter("tb/" + name, sess.graph)

    # Find the explanation
    points_rep = sess.run(rep, feed_dict={X: points})
    plt.scatter(points_rep[:,0], points_rep[:,1], marker = "v", c = "violet", s = 64)

    for i in range(500): #TODO:  stopping condition
        grad = sess.run([new_global, new_ind], feed_dict={X: points, T: target})
        if i % 10 == 0:
            summary = sess.run(summary_op, feed_dict={X: points, T: target})
            writer.add_summary(summary, i)

    points_rep = sess.run(rep, feed_dict={X: points})
    plt.scatter(points_rep[:,0], points_rep[:,1], marker = "^", c = "violet", s = 64)

    plt.subplot(2,1,2)

    d_g, d_i = sess.run([delta_global, delta_ind])
    axis = np.array(range(x.shape[1]))

    plt.scatter(axis, d_g, label = "Delta Global", marker = "x")
    plt.scatter(axis, np.mean(d_i, axis = 0), label = "Delta Individual - Mean", marker = "1")
    plt.scatter(axis, np.var(d_i, axis = 0), label = "Delta Individual - Var", marker = "2")

    plt.legend(fancybox=True, framealpha=0.5)

    plt.show()
    
    plt.close()
    
    return d_g, d_i


