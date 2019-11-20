
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

import sys
sys.path.insert(0, "/home/gregory/Desktop/ELDR/scvis/lib/scvis/")
from vae import GaussianVAE

def load_vae(input_dim, model_file, feature_transform = None):

    tf.reset_default_graph()
    
    # Model Configuration
    try:
        config_file_yaml = open("/home/gregory/Desktop/ELDR/scvis/lib/scvis/config/model_config.yaml", "r")
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
