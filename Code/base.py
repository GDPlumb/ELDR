
import numpy as np
import tensorflow as tf

class MLP():
    # shape[0] = input dimension
    # shape[end] = output dimension
    def __init__(self, shape):
        self.shape = shape
        self.weight_init = tf.contrib.layers.xavier_initializer()
        self.bias_init = tf.constant_initializer(0.0)
    
    def layer(self, input, input_size, output_size):
        weights = tf.get_variable("weights", [input_size, output_size], initializer = self.weight_init)
        biases = tf.get_variable("biases", output_size, initializer = self.bias_init)
        return tf.nn.leaky_relu(tf.matmul(input, weights) + biases)

    def model(self, input):
        shape = self.shape
        n = len(shape)
        x = input
        for i in range(n - 2):
            with tf.variable_scope("hidden_" + str(i + 1)):
                out = self.layer(x, shape[i], shape[i + 1])
                x = out
        with tf.variable_scope("output"):
            weights = tf.get_variable("weights", [shape[n - 2], shape[n - 1]], initializer = self.weight_init)
            biases = tf.get_variable("biases", shape[n - 1], initializer = self.bias_init)
            #return tf.squeeze(tf.matmul(x, weights) + biases)
            return tf.matmul(x, weights) + biases

def load_encoder(input_dim, model_file, encoder_shape = [100, 100, 100, 2]):

    # Reset TF graph (avoids issues with repeat experiments)
    tf.reset_default_graph()

    # Setup our modified input to the model
    X = tf.placeholder(tf.float32, shape=[None, input_dim])
    D = tf.placeholder(tf.float32, shape=[1, input_dim])
    
    input = X + D
    
    # Build the models
    shape = encoder_shape.copy()
    shape.insert(0, input_dim)
    encoder = MLP(shape)
    with tf.variable_scope("encoder_model", reuse = tf.AUTO_REUSE):
        rep = encoder.model(input)

    # Setup and restore the tf session
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    return sess, rep, X, D

# Source: https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
# Modification: added a response variable to the dataset
class BatchManager():

    def __init__(self, X, Y):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._X = X
        self._num_examples = X.shape[0]
        self._Y = Y

    @property
    def X(self):
        return self._X
    
    @property
    def Y(self):
        return self._Y

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        
        # Shuffle the data on the first call
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)
            np.random.shuffle(idx)
            self._X = self.X[idx]
            self._Y = self.Y[idx]
        
        # If there aren't enough points left in this epoch to fill the minibatch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start

            # Load the remaining data
            X_rest_part = self.X[start:self._num_examples]
            Y_rest_part = self.Y[start:self._num_examples]
            
            # Reshuffle the dataset
            idx0 = np.arange(0, self._num_examples)
            np.random.shuffle(idx0)
            self._X = self.X[idx0]
            self._Y = self.Y[idx0]
            
            # Get the remaining samples for the batch from the next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end =  self._index_in_epoch
            X_new_part = self._X[start:end]
            Y_new_part = self._Y[start:end]
            return np.concatenate((X_rest_part, X_new_part), axis=0), np.concatenate((Y_rest_part, Y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._X[start:end], self._Y[start:end]



