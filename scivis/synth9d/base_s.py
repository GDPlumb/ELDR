import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

import sys
sys.path.insert(0, "../scvis-dev/lib/scvis/")
from vae import GaussianVAE

from base import pdist

def load_vae_s(num_points):
    tf.reset_default_graph()
    
    try:
        config_file_yaml = open("../scvis-dev/lib/scvis/config/model_config.yaml", "r")
        config = yaml.load(config_file_yaml, Loader = yaml.FullLoader)
        config_file_yaml.close()
    except yaml.YAMLError as exc:
        print("Error in the configuration file: {}".format(exc))

    architecture = config["architecture"]
    architecture.update({"input_dimension": 9})

    X = tf.placeholder(tf.float32, shape = [None, 9])
    D = tf.placeholder(tf.float32, shape = [1, 1])

    delta = tf.Variable(tf.zeros(shape = [1, 9]), name = "delta")
    delta_d1 = tf.Variable(tf.zeros(shape = [num_points, 9]), name = "delta_d1")
    delta_d2 = tf.Variable(tf.zeros(shape = [num_points, 9]), name = "delta_d2")

    input = X + D * delta + 0.5 * (D + 1.0) * delta_d1 + 0.5 * (D - 1.0) * delta_d2

    vae = GaussianVAE(input, 1, architecture['inference']['layer_size'], architecture['latent_dimension'], decoder_layer_size=architecture['model']['layer_size'])
    rep, _ = vae.encoder(prob = 1.0)

    sess = tf.Session()

    sess.run(delta.initializer)
    sess.run(delta_d1.initializer)
    sess.run(delta_d2.initializer)

    old_vars = tf.trainable_variables()[3:]
    saver = tf.train.Saver(var_list = old_vars)
    saver.restore(sess, "model/model/perplexity_10_regularizer_0.001_batch_size_512_learning_rate_0.01_latent_dimension_2_activation_ELU_seed_1_iter_3000.ckpt")

    return sess, rep, X, D, delta, delta_d1, delta_d2

def explain_s(x, y, indices, c1, c2, num_points = 20, dispersion = 1.5, lambda_global = 0.5, lambda_ind = 4):
    name = str(c1) + "to" + str(c2)
    
    # Visualize the data
    fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(2,1,1)
    plt.scatter(y[:, 0], y[:, 1], s = 12)
    
    # Sample num_points in clusters c1 and c2 from x
    points_c1 = x[np.random.choice(indices[c1], num_points, replace = False)]
    points_c2 = x[np.random.choice(indices[c2], num_points, replace = False)]
    
    # The target point is the center of the other cluster in y
    target_c1 = np.mean(y[indices[c1], :], axis = 0).reshape((1,2))
    target_c2 = np.mean(y[indices[c2], :], axis = 0).reshape((1,2))

    # Define the objective function
    sess, rep, X, D, delta, delta_d1, delta_d2 = load_vae_s(num_points)

    T = tf.placeholder(tf.float32, shape=[None, 2])
    l_t = tf.losses.mean_squared_error(T, rep)
    tf.summary.scalar("loss/t", l_t)
    l_d = dispersion * tf.reduce_mean(1 / (1 + pdist(rep)))
    tf.summary.scalar("loss/d", l_d)
    l_g = lambda_global * tf.reduce_mean(tf.abs(delta))
    tf.summary.scalar("loss/g", l_g)
    l_i = lambda_ind * tf.reduce_mean(tf.abs(0.5 * (D + 1.0) * delta_d1 + 0.5 * (D - 1.0) * delta_d2))
    tf.summary.scalar("loss/i", l_i)

    loss = l_t + l_d + l_g + l_i
    grad = tf.gradients(loss, [delta, delta_d1, delta_d2])
    new_global = delta.assign(delta - 0.05 * tf.clip_by_value(grad[0], -10.0, 10.0))
    new_d1 = delta_d1.assign(delta_d1 - 0.05 * tf.clip_by_value(grad[1], -10.0, 10.0))
    new_d2 = delta_d2.assign(delta_d2 - 0.05 * tf.clip_by_value(grad[2], -10.0, 10.0))
    
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter("tb/" + name, sess.graph)

    # Find the explanation
    d = np.ones((1,1))
    y_c1 = sess.run(rep, feed_dict={X: points_c1, D: d})
    y_c2 = sess.run(rep, feed_dict={X: points_c2, D: -1.0 * d})

    plt.scatter(y_c1[:,0], y_c1[:,1], marker = "v", c = "violet", s = 64)
    plt.scatter(y_c2[:,0], y_c2[:,1], marker = "v", c = "red", s = 64)

    for i in range(1001): #TODO:  stopping condition

        if d[0] == 1.0:
            dict = {X: points_c1, T: target_c2, D: d}
            sess.run([new_global, new_d1], feed_dict = dict)
        else:
            dict = {X: points_c2, T: target_c1, D: d}
            sess.run([new_global, new_d2], feed_dict = dict)
    
        if i % 25 == 0:
            summary = sess.run(summary_op, feed_dict = dict)
            writer.add_summary(summary, i)

        d *= -1.0
        
    writer.flush() 
    
    d = np.ones((1,1))
    y_c1 = sess.run(rep, feed_dict={X: points_c1, D: d})
    y_c2 = sess.run(rep, feed_dict={X: points_c2, D: -1.0 * d})

    plt.scatter(y_c1[:,0], y_c1[:,1], marker = "^", c = "violet", s = 64)
    plt.scatter(y_c2[:,0], y_c2[:,1], marker = "^", c = "red", s = 64)

    plt.subplot(2,1,2)

    d_g, d_d1, d_d2 = sess.run([delta, delta_d1, delta_d2])
    axis = np.array(range(x.shape[1]))

    plt.scatter(axis, d_g, label = "Delta Global", marker = "x")
    plt.scatter(axis, np.mean(d_d1, axis = 0), label = "Delta D1 - Mean", marker = "1")
    plt.scatter(axis, np.var(d_d1, axis = 0), label = "Delta D1 - Var", marker = "2")
    plt.scatter(axis, np.mean(d_d2, axis = 0), label = "Delta D2 - Mean", marker = "1")
    plt.scatter(axis, np.var(d_d2, axis = 0), label = "Delta D2 - Var", marker = "2")

    plt.legend(fancybox=True, framealpha=0.5)

    plt.show()
    
    plt.close()
    
    return d_g, d_d1, d_d2
