
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
import sys
import tensorflow as tf

from base import  MLP, BatchManager

def train_ae(x,
          encoder_shape = [100, 100, 100, 2], decoder_shape = [2, 100, 100, 100],
          learning_rate = 0.001, batch_size = 4, min_epochs = 100, stopping_epochs = 50, tol = 0.001, freq_eval = 1):
    
    # Allow multiple sessions on a single GPU.
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # Reset TF graph (avoids issues with repeat experiments)
    tf.reset_default_graph()
    
    # Setup directory
    os.system("rm -rf Model")
    cwd = os.getcwd()
    os.makedirs("Model")
    os.chdir("Model")
    
    sys.stdout = open("train.txt", "w")
    
    # Split the dataset
    x_train, x_val = train_test_split(x, test_size = 0.25)
    
    # Get sizes for future reference
    n = x_train.shape[0]
    n_input = x_train.shape[1]
    encoder_shape.insert(0, n_input)
    decoder_shape.append(n_input)
    
    # Batch Manager
    y_train = np.zeros((n, 1)) #Dumby variable to let use the BatchManager
    bm = BatchManager(x_train, y_train)

    # Graph inputs
    X = tf.placeholder("float", [None, n_input], name = "X_in")
    R = tf.placeholder("float", [None, 2], name = "R_in")

    # Build the models
    encoder = MLP(encoder_shape)
    with tf.variable_scope("encoder_model", reuse = tf.AUTO_REUSE):
        rep = encoder.model(X)
    
    decoder = MLP(decoder_shape)
    with tf.variable_scope("decoder_model", reuse = tf.AUTO_REUSE):
        recon = decoder.model(rep)

    # Define the loss and optimizer
    recon_loss = tf.losses.mean_squared_error(labels = X, predictions = recon)
    tf.summary.scalar("ReconMSE", recon_loss)
    
    loss_op = recon_loss
    tf.summary.scalar("Loss", loss_op)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss_op)

    summary_op = tf.summary.merge_all()

    # Train and evaluate the model
    init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    saver = tf.train.Saver(max_to_keep = 1)

    best_epoch = 0
    best_loss = np.inf

    with tf.Session(config = tf_config) as sess:
        train_writer = tf.summary.FileWriter("train", sess.graph)
        val_writer = tf.summary.FileWriter("val")

        sess.run(init)
        epoch = 0
        total_batch = int(n / batch_size)
        while True:

            # Stopping condition
            if epoch - best_epoch > stopping_epochs and epoch > min_epochs:
                break

            # Run a training epoch
            for i in range(total_batch):
                x_batch, y_batch = bm.next_batch(batch_size = batch_size)
                summary, _ = sess.run([summary_op, train_op], feed_dict = {X: x_batch})
                train_writer.add_summary(summary, epoch * total_batch + i)

            # Run model metrics
            if epoch % freq_eval == 0:
                
                summary, val_loss = sess.run([summary_op, loss_op], feed_dict = {X: x_val})
                
                if val_loss < best_loss - tol:
                    print(epoch, " ", val_loss)
                    best_loss = val_loss
                    best_epoch = epoch
                    saver.save(sess, "./model.cpkt")
        
                val_writer.add_summary(summary, (epoch + 1) * total_batch)

            epoch += 1

        train_writer.close()
        val_writer.close()

        # Evaluate the final model
        saver.restore(sess, "./model.cpkt")

        # Find the 2d point representation
        points = np.zeros((n, 2))
        for i in range(total_batch):
            start = i * batch_size
            stop = min(n, (i + 1) * batch_size)
            x_batch = x[start:stop, :]
            points_batch = sess.run(rep, {X: x_batch})
            points[start:stop, :] = points_batch
        plt.scatter(points[:, 0], points[:, 1], s = 10)
        plt.savefig("representation.pdf")
        plt.close()
        pickle.dump(points, open("points.pkl", "wb"))

        # Go back to directory
        os.chdir(cwd)
