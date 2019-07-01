import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

def explain(load_model, x, y, indices, c1, c2, num_points = 20, dispersion = 1.5, lambda_global = 0.5, lambda_ind = 4):
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
    sess, rep, X, delta_global, delta_ind = load_model(num_points)

    T = tf.placeholder(tf.float32, shape=[None, 2])
    l_t = tf.losses.mean_squared_error(T, rep)
    tf.summary.scalar("loss/target", l_t)
    l_d = dispersion * tf.reduce_mean(1 / (1 + pdist(rep)))
    tf.summary.scalar("loss/dispersion", l_d)
    l_g = lambda_global * tf.reduce_mean(tf.abs(delta_global))
    tf.summary.scalar("loss/global", l_g)
    l_i = lambda_ind * tf.reduce_mean(tf.abs(delta_ind))
    tf.summary.scalar("loss/individual", l_i)

    loss = l_t + l_d + l_g + l_i
    tf.summary.scalar("loss/total", loss)
    grad = tf.gradients(loss, [delta_global, delta_ind])
    new_global = delta_global.assign(delta_global - 0.01 * grad[0])
    new_ind = delta_ind.assign(delta_ind - 0.01 * grad[1])
    
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter("tb/" + name, sess.graph)

    # Find the explanation
    points_rep = sess.run(rep, feed_dict={X: points})
    plt.scatter(points_rep[:,0], points_rep[:,1], marker = "v", c = "violet", s = 64)

    for i in range(1000): #TODO:  stopping condition
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

    plt.legend(fancybox=True, framealpha = 0.5)

    plt.show()
    
    plt.close()
    
    return d_g, d_i

def explain_sym(load_model, x, y, indices, c1, c2, num_points = 30, dispersion_c1 = 1.5, dispersion_c2 = 1.5, lambda_global = 0.5, lambda_ind = 4.0):
    name = str(c1) + "to" + str(c2)
    
    # Visualize the data
    fig, ax = plt.subplots(figsize=(20, 10))

    plt.subplot(2,1,1)
    plt.scatter(y[:, 0], y[:, 1], s = 12)
    
    # Sample num_points in clusters c1 and c2 from x and y
    indices_c1 = np.random.choice(indices[c1], num_points, replace = False)
    indices_c2 = np.random.choice(indices[c2], num_points, replace = False)

    points_c1 = x[indices_c1]
    points_c2 = x[indices_c2]
    
    targets_c1 = y[indices_c1]
    targets_c2 = y[indices_c2]

    # Define the objective function
    sess, rep, X, D, delta, delta_d1, delta_d2 = load_model(num_points)

    T = tf.placeholder(tf.float32, shape=[None, 2])
    l_t = tf.reduce_mean(tf.reduce_min(pairwise_l2_norm2(rep, T), axis = 1))
    tf.summary.scalar("loss/target", l_t)
    l_d = (dispersion_c1 * -0.5 * (D[0,0] - 1.0) + dispersion_c2 * 0.5 * (D[0,0] + 1.0)) * tf.reduce_mean(1 / (1 + pdist(rep)))
    tf.summary.scalar("loss/dispersion", l_d)
    l_g = lambda_global * tf.reduce_mean(tf.abs(delta))
    tf.summary.scalar("loss/global", l_g)
    l_i = lambda_ind * tf.reduce_mean(tf.abs(0.5 * (D[0,0] + 1.0) * delta_d1 + 0.5 * (D[0,0] - 1.0) * delta_d2))
    tf.summary.scalar("loss/individual", l_i)

    loss = l_t + l_d + l_g + l_i
    tf.summary.scalar("loss/total", loss)
    grad = tf.gradients(loss, [delta, delta_d1, delta_d2])
    new_global = delta.assign(delta - 0.001 * tf.clip_by_value(grad[0], -5.0, 5.0))
    new_d1 = delta_d1.assign(delta_d1 - 0.001 * tf.clip_by_value(grad[1], -5.0, 5.0))
    new_d2 = delta_d2.assign(delta_d2 - 0.001 * tf.clip_by_value(grad[2], -5.0, 5.0))
    
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter("tb/" + name, sess.graph)

    # Find the explanation
    d = np.ones((1,1))
    y_c1 = sess.run(rep, feed_dict={X: points_c1, D: d})
    y_c2 = sess.run(rep, feed_dict={X: points_c2, D: -1.0 * d})

    plt.scatter(y_c1[:,0], y_c1[:,1], marker = "v", c = "violet", s = 64)
    plt.scatter(y_c2[:,0], y_c2[:,1], marker = "v", c = "red", s = 64)

    for i in range(4001): #TODO:  stopping condition

        if d[0] == 1.0:
            dict = {X: points_c1, T: targets_c2, D: d}
            sess.run([new_global, new_d1], feed_dict = dict)
        else:
            dict = {X: points_c2, T: targets_c1, D: d}
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

