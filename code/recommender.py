# COMP5329 - Deep Learning
# Assignment 2
# Authors: King Tao Ng and Anjali Upadhyay
# recommender.py: Main module

from glm import GLM
from neuralnetwork import NeuralNetwork
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import h5py

class Recommender:

    # Read file into pandas
    def read_data(self, file):
        """
        file: source of data
        """
        data = pd.read_csv(file)
        data = data.sample(frac=1) # shuffle randomly
        # coauthorship is a count we want to predict. distance, is_same_title, is_same_department, duration_difference are independent variables.
        subset = data[["coauthorship", "distance", "is_same_title", "is_same_department", "duration_difference"]]
        return subset
    
    # K-fold cross validation
    def cross_validation(self, subset, k_fold, is_randomised = True):
        """
        subset: source of data
        k_fold: a number of folds
        is_randomised: randomly shuffle subset if is_randomised = True
        """
        cv = KFold(n_splits=k_fold, shuffle=is_randomised)
        for train_index, test_index in cv.split(subset):
            training_data, test_data = subset.iloc[train_index], subset.iloc[test_index]
            yield training_data, test_data
    
    # Poisson loss function
    def poisson_loss(self, truth, predicted):
        """
        truth: ground truth
        predicted: predicted count
        """
        epsilon = 1e-9 # avoid log 0
        return np.sum(predicted - truth * np.log(predicted + epsilon), axis=0)/predicted.size

    # Poisson regression (i.e. baseline). Some situations cause this model to freeze. Restart an experiment if you encounter situations.
    def poisson(self, training_data, test_data):
        """
        training_data: data used for training
        test_data: data used for testing
        """
        glm = GLM("""coauthorship ~ distance + is_same_title + is_same_department + duration_difference""")
        glm.poisson(training_data)
        summary, predicted = glm.predict(test_data)
        predicted[predicted < 0] = 0 # Indentity link would return negatives, which are mapped to 0.
        loss = self.poisson_loss(predicted, test_data["coauthorship"])
        return predicted, loss

    # Negative Binomial regression. Poisson regression suffers overdispersion and underdispersion.
    def negative_binomial(self, training_data, test_data):
        """
        training_data: data used for training
        test_data: data used for testing
        """
        glm = GLM("""coauthorship ~ distance + is_same_title + is_same_department + duration_difference""")
        glm.negative_binomial(training_data, alpha=0.01)
        summary, predicted = glm.predict(test_data)
        predicted[predicted < 0] = 0 # Indentity link would return negatives, which are mapped to 0.
        loss = self.poisson_loss(predicted, test_data["coauthorship"])
        return predicted, loss

    # Neural network
    def neuralnetwork(self, training_data, test_data):
        """
        training_data: data used for training
        test_data: data used for testing
        """
        neuralnetwork = NeuralNetwork(weights = 10, learning_rate = 0.001, epochs = 100000)
        neuralnetwork.neuralnetwork(training_data)
        predicted = neuralnetwork.predict(test_data)
        loss = self.poisson_loss(predicted, test_data["coauthorship"])
        return predicted, loss
    
    # Retrieve weights and bias from experiments. They are used for coefficient credibility.
    def get_neuralnetwork_weights(self, training_data, test_data):
        neuralnetwork = NeuralNetwork(weights = 10, learning_rate = 0.001, epochs = 100000)
        neuralnetwork.neuralnetwork(training_data)
        weights, bias = neuralnetwork.get_weights_and_bias()
        return weights, bias

    # Get the posterior distributions
    def get_neuralnetwork_posterior(self, test_data, weights, bias):
        neuralnetwork = NeuralNetwork()
        neuralnetwork_predicted = neuralnetwork.get_posterior_distributions(test_data, weights, bias)
        return neuralnetwork_predicted

    def print_neuralnetwork(self, filename, test_data, predicted_mean, predicted_min, predicted_max):
        coauthorship = test_data[["coauthorship"]]
        with open(filename, mode='w') as file:
            counter = 0
            file.write("Truth,Neural Network(Mean),Neural Network(Min),Neural Network(Max), Fall?\n")
            for i in range(len(coauthorship)):
                truth = test_data.iloc[i]['coauthorship']
                counter += predicted_min[i] <= truth and truth <= predicted_max[i]
                file.write(str(truth) + "," + str(predicted_mean[i]) + "," + str(predicted_min[i]) + "," + str(predicted_max[i]) + "," + str(predicted_min[i] <= truth and truth <= predicted_max[i]) + "\n")
            #print(counter/len(coauthorship))
                
    # Print results into file
    def print_results(self, filename, test_data, poisson_predicted, negative_binomial_predicted, neuralnetwork_predicted):
        """
        filename: a file where results are written into
        test_data: ground truth
        poisson_predicted: results predicted from poisson
        negative_binomial_predicted: results predicted from negative binomial
        neuralnetwork_predicted: results predicted from neural network
        """
        coauthorship = test_data[["coauthorship"]]
        poisson_predicted = poisson_predicted.round(0) # round to an integer
        negative_binomial_predicted = negative_binomial_predicted.round(0) # round to an integer
        neuralnetwork_predicted = np.round(neuralnetwork_predicted, 0) # round to an integer
        
        poisson_mse = mean_squared_error(coauthorship, poisson_predicted) # computer MSE
        negative_binomial_mse = mean_squared_error(coauthorship, negative_binomial_predicted) # compute MSE
        neuralnetwork_mse = mean_squared_error(coauthorship, neuralnetwork_predicted) # compute MSE
        
        with open(filename, mode='w') as file:
            file.write("Truth, Poisson, Negative Binomial, Neural Network\n")
            for i in range(len(coauthorship)):
                truth = test_data.iloc[i]['coauthorship']
                poisson = poisson_predicted.iloc[i]
                negative_binomial = negative_binomial_predicted.iloc[i]
                neuralnetwork = neuralnetwork_predicted[i]
                file.write(str(truth) + "," + str(poisson) + "," + str(negative_binomial) + "," + str(neuralnetwork) + "\n")
            file.write("MSE," + str(poisson_mse) + "," + str(negative_binomial_mse) + "," + str(neuralnetwork_mse))
            print("Results written into " + filename)

    # Plot weights and bias into histograms
    def plot_histograms(self, training_weights, training_bias):
        bin_size = 100
        
        beta0 = training_bias.flatten()
        beta1 = training_weights[0]
        beta2 = training_weights[1]
        beta3 = training_weights[2]
        beta4 = training_weights[3]
        
        fig, axs = plt.subplots(2, 3, sharey=True, tight_layout=True)
        
        axs[0][0].hist(beta0, bins=bin_size)
        axs[0][1].hist(beta1, bins=bin_size)
        axs[0][2].hist(beta2, bins=bin_size)
        axs[1][0].hist(beta3, bins=bin_size)
        axs[1][1].hist(beta4, bins=bin_size)
        
        plt.show()
    
if __name__ == "__main__":
    
    recommender = Recommender()
    subset = recommender.read_data("../data/data.csv")
    
    # Build models and predict unseen data, and results are saved down into the "../data/" folder
    if False:
        counter = 1
        for training_data, test_data, in recommender.cross_validation(subset, k_fold=10):
            poisson_predicted, poisson_loss = recommender.poisson(training_data, test_data)
            negative_binomial_predicted, negative_binomial_loss = recommender.negative_binomial(training_data, test_data)
            neuralnetwork_predicted, neuralnetwork_loss = recommender.neuralnetwork(training_data, test_data)
            recommender.print_results("../data/result" + str(counter) + ".csv", test_data, poisson_predicted, negative_binomial_predicted, neuralnetwork_predicted)
            counter += 1
    
    # Get weight and bias distributions
    if False:
        training_weights = None
        training_bias = None
        for i in range(100):
            print("Running the " + str(i + 1) + "th experiment...")
            for training_data, test_data, in recommender.cross_validation(subset, k_fold=10):
                weights, bias = recommender.get_neuralnetwork_weights(training_data, test_data)
                training_weights = weights if training_weights is None else np.append(training_weights, weights, axis = 1)
                training_bias = bias if training_bias is None else np.append(training_bias, bias, axis = 1)

        # Plot weights and bias into histograms
        recommender.plot_histograms(training_weights, training_bias)

        # Save weights and bias for posterior distributions
        with h5py.File('../data/weights.hdf5', 'w') as f:
            weights = f.create_dataset("weights", data=training_weights)
        with h5py.File('../data/bias.hdf5', 'w') as f:
            bias = f.create_dataset("bias", data=training_bias)
    
    # Get posterior for the first four unseen observations
    if False:
        with h5py.File('../data/weights.hdf5','r') as f:
            weights = np.copy(f["weights"])
        with h5py.File('../data/bias.hdf5','r') as f:
            bias = np.copy(f["bias"])
        more_than_one = subset[subset["coauthorship"] > 1]
        neuralnetwork_predicted = recommender.get_neuralnetwork_posterior(more_than_one, weights, bias)

        print(more_than_one.iloc[0,:])
        print(more_than_one.iloc[1,:])
        print(more_than_one.iloc[2,:])
        print(more_than_one.iloc[3,:])
        
        fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
        
        axs[0][0].hist(neuralnetwork_predicted[0,:], bins=100)
        axs[0][1].hist(neuralnetwork_predicted[1,:], bins=100)
        axs[1][0].hist(neuralnetwork_predicted[2,:], bins=100)
        axs[1][1].hist(neuralnetwork_predicted[3,:], bins=100)
        
        plt.show()

    # Get posterior distributions
    if True:
        with h5py.File('../data/weights.hdf5','r') as f:
            weights = np.copy(f["weights"])
        with h5py.File('../data/bias.hdf5','r') as f:
            bias = np.copy(f["bias"])
        counter = 1
        for training_data, test_data, in recommender.cross_validation(subset, k_fold=10):
            neuralnetwork_predicted = recommender.get_neuralnetwork_posterior(test_data, weights, bias)
            predicted_mean = np.mean(neuralnetwork_predicted, axis=1)
            predicted_max = np.max(neuralnetwork_predicted, axis=1)
            predicted_min = np.min(neuralnetwork_predicted, axis=1)
            recommender.print_neuralnetwork("../data/posterior" + str(counter) + ".csv", test_data, predicted_mean, predicted_min, predicted_max)
            counter += 1
