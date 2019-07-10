
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from explain_normal import pdist

print("")
print("WARNING:  NEEDS TO BE UPDATED")
print("")


def load_vae_sym(input_dim, model_file, num_points, feature_transform = None):
    tf.reset_default_graph()
    
    try:
        config_file_yaml = open("../scvis/lib/scvis/config/model_config.yaml", "r")
        config = yaml.load(config_file_yaml, Loader = yaml.FullLoader)
        config_file_yaml.close()
    except yaml.YAMLError as exc:
        print("Error in the configuration file: {}".format(exc))

    architecture = config["architecture"]
    architecture.update({"input_dimension": input_dim})

    X = tf.placeholder(tf.float32, shape = [None, input_dim])
    D = tf.placeholder(tf.float32, shape = [1, 1])

    delta = tf.Variable(tf.zeros(shape = [1, input_dim]), name = "delta")
    delta_d1 = tf.Variable(tf.zeros(shape = [num_points, input_dim]), name = "delta_d1")
    delta_d2 = tf.Variable(tf.zeros(shape = [num_points, input_dim]), name = "delta_d2")

    input = X + D * delta + 0.5 * (D[0,0] + 1.0) * delta_d1 + 0.5 * (D[0,0] - 1.0) * delta_d2
    
    if feature_transform is not None:
        matrix =  np.float32(pd.read_csv(feature_transform, sep="\t", header = None).values)
        input = tf.matmul(input, matrix)
    
    vae = GaussianVAE(input, 1, architecture["inference"]["layer_size"], architecture["latent_dimension"], decoder_layer_size=architecture["model"]["layer_size"])
    rep, _ = vae.encoder(prob = 1.0)

    sess = tf.Session()

    sess.run(delta.initializer)
    sess.run(delta_d1.initializer)
    sess.run(delta_d2.initializer)

    old_vars = tf.trainable_variables()[3:]
    saver = tf.train.Saver(var_list = old_vars)
    saver.restore(sess, model_file)

    return sess, rep, X, D, delta, delta_d1, delta_d2

def explain_sym(load_model, x, y, indices, c1, c2,
                num_points = 100, dispersion_c1 = 1.5, dispersion_c2 = 1.5, lambda_global = 0.5, lambda_ind = 4.0,
                learning_rate = 0.001, grad_clip = 5.0, min_epochs = 500, stopping_epochs = 500, tol = 0.01):
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

    loss_op = l_t + l_d + l_g + l_i
    tf.summary.scalar("loss/total", loss_op)
    grad = tf.gradients(loss_op, [delta, delta_d1, delta_d2])
    new_global = delta.assign(delta - learning_rate * tf.clip_by_value(grad[0], -1.0 * grad_clip, grad_clip))
    new_d1 = delta_d1.assign(delta_d1 - learning_rate * tf.clip_by_value(grad[1], -1.0 * grad_clip, grad_clip))
    new_d2 = delta_d2.assign(delta_d2 - learning_rate * tf.clip_by_value(grad[2], -1.0 * grad_clip, grad_clip))
    
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter("tb/" + name, sess.graph)

    # Find the explanation
    d = np.ones((1,1))
    y_c1 = sess.run(rep, feed_dict={X: points_c1, D: d})
    y_c2 = sess.run(rep, feed_dict={X: points_c2, D: -1.0 * d})

    plt.scatter(y_c1[:,0], y_c1[:,1], marker = "v", c = "violet", s = 64)
    plt.scatter(y_c2[:,0], y_c2[:,1], marker = "v", c = "red", s = 64)
    
    best_epoch = 0
    best_loss = np.inf
    saver = tf.train.Saver(max_to_keep = 1)
    epoch = 0
    while True:
    
        if epoch > min_epochs and epoch - best_epoch > stopping_epochs:
            break

        if d[0] == 1.0:
            dict = {X: points_c1, T: targets_c2, D: d}
            sess.run([new_global, new_d1], feed_dict = dict)
        else:
            dict = {X: points_c2, T: targets_c1, D: d}
            sess.run([new_global, new_d2], feed_dict = dict)
    
        if epoch % 5 == 0:
            summary, loss = sess.run([summary_op, loss_op], feed_dict = dict)
            writer.add_summary(summary, epoch)
        
            if loss < best_loss - tol:
                best_loss = loss
                best_epoch = epoch
                saver.save(sess, "./model.cpkt")

        d *= -1.0
        epoch += 1
    
    writer.flush()
    
    saver.restore(sess, "./model.cpkt")
    
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
    
    return np.ndarray.flatten(d_g), d_d1, d_d2


