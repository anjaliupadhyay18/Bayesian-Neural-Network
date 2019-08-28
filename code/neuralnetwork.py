# COMP5329 - Deep Learning
# Assignment 2
# Authors: King Tao Ng and Anjali Upadhyay
# neuralnetwork: It aims to train a neural network.

import tensorflow as tf

class NeuralNetwork:

    # Initialise a neural network
    def __init__(self, weights = 10, learning_rate = 0.001, epochs = 100000):
        self.weights = weights
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    # Train neural network
    def neuralnetwork(self, training_data):
        training_X = training_data[["distance", "is_same_title", "is_same_department", "duration_difference"]].values
        training_Y = training_data[["coauthorship"]]

        X = tf.constant(value = training_X, name="X")
        X = tf.cast(X, tf.float64)
        y = tf.constant(value = training_Y, name="y")
        y = tf.cast(y, tf.float64)
        
        self.b = tf.Variable(tf.zeros([1, 1]), name="bias")
        self.b = tf.cast(self.b, tf.float64)
        self.w = tf.Variable(tf.ones([4, 1]), name="weights")
        self.w = tf.cast(self.w, tf.float64)
        
        y_hat = tf.nn.relu(tf.matmul(X, self.w) + self.b)
        loss_function = tf.reduce_mean(tf.losses.mean_squared_error(y, y_hat, weights=self.weights))
        
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_function)
        self.init = tf.global_variables_initializer()
    
    # Predict unseen data
    def predict(self, test_data):
        session = tf.InteractiveSession()
        session.run(self.init)
        for i in range(self.epochs):
            session.run(self.train_step)
        
        test_X = test_data[["distance", "is_same_title", "is_same_department", "duration_difference"]].values
        X = tf.constant(value = test_X, name = "X")
        X = tf.cast(X, tf.float64)
        feed_dict = {X: test_X}
        
        y_hat = tf.nn.relu(tf.matmul(X, self.w) + self.b)
        predicted = session.run(y_hat, feed_dict)
        predicted = predicted.reshape(len(predicted),)
        
        session.close() # prevent memory leak
        return predicted
    
    # Retrieve weights and bias
    def get_weights_and_bias(self):
        session = tf.InteractiveSession()
        session.run(self.init)
        for i in range(self.epochs):
            session.run(self.train_step)
        
        weights = session.run(self.w)
        bias = session.run(self.b)
        session.close()
        return weights, bias
    
    def get_posterior_distributions(self, test_data, weights, bias):
        test_X = test_data[["distance", "is_same_title", "is_same_department", "duration_difference"]].values
        X = tf.placeholder(tf.float64, shape=(test_X.shape[0], 4))
        
        w = tf.constant(value = weights, name="weights")
        b = tf.constant(value = bias, name="bias")

        y_hat = tf.nn.relu(tf.matmul(X, w) + b)
        init = tf.global_variables_initializer()

        feed_dict = {X: test_X}
        session = tf.InteractiveSession()
        session.run(init)
        predicted = session.run(y_hat, feed_dict)
        session.close()
        return predicted
