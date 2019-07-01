
import tensorflow as tf
import yaml

import sys
sys.path.insert(0, "../scvis-dev/lib/scvis/")
from vae import GaussianVAE

def load_original(input_dim, model_file):
    tf.reset_default_graph()
    
    try:
        config_file_yaml = open("../scvis-dev/lib/scvis/config/model_config.yaml", "r")
        config = yaml.load(config_file_yaml, Loader = yaml.FullLoader)
        config_file_yaml.close()
    except yaml.YAMLError as exc:
        print("Error in the configuration file: {}".format(exc))

    architecture = config["architecture"]
    architecture.update({"input_dimension": input_dim})

    X = tf.placeholder(tf.float32, shape=[None, input_dim])

    vae = GaussianVAE(X, 1, architecture["inference"]["layer_size"], architecture["latent_dimension"], decoder_layer_size=architecture["model"]["layer_size"])
    rep, _ = vae.encoder(prob = 1.0)

    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    return sess, rep, X

def load_vae(input_dim, model_file, num_points):
    tf.reset_default_graph()
    
    try:
        config_file_yaml = open("../scvis-dev/lib/scvis/config/model_config.yaml", "r")
        config = yaml.load(config_file_yaml, Loader = yaml.FullLoader)
        config_file_yaml.close()
    except yaml.YAMLError as exc:
        print("Error in the configuration file: {}".format(exc))

    architecture = config["architecture"]
    architecture.update({"input_dimension": input_dim})

    X = tf.placeholder(tf.float32, shape=[None, input_dim])
    delta_global = tf.Variable(tf.zeros(shape = [1, input_dim]), name = "delta_global")
    delta_ind = tf.Variable(tf.zeros(shape = [num_points, input_dim]), name = "delta_ind")
    input = X + delta_global + delta_ind

    vae = GaussianVAE(input, 1, architecture["inference"]["layer_size"], architecture["latent_dimension"], decoder_layer_size=architecture["model"]["layer_size"])
    rep, _ = vae.encoder(prob = 1.0)

    sess = tf.Session()

    sess.run(delta_global.initializer)
    sess.run(delta_ind.initializer)
    old_vars = tf.trainable_variables()[2:]
    saver = tf.train.Saver(var_list = old_vars)
    saver.restore(sess, model_file)

    return sess, rep, X, delta_global, delta_ind

def load_vae_sym(input_dim, model_file, num_points):
    tf.reset_default_graph()
    
    try:
        config_file_yaml = open("../scvis-dev/lib/scvis/config/model_config.yaml", "r")
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
