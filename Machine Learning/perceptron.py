import numpy as np

import backend

class Perceptron(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.get_data_and_monitor = backend.make_get_data_and_monitor_perceptron()

        self.weights = np.zeros(dimensions) #creates weight vector of zeros with input dimensions

    def get_weights(self):
        """
        Return the current weights of the perceptron.

        Returns: a numpy array with D elements, where D is the value of the
            `dimensions` parameter passed to Perceptron.__init__
        """
        
        return self.weights

    def predict(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """

        if np.dot(self.get_weights(), x) >= 0: #if nonnegative return 1 else -1
            return 1
        else:
            return -1

    def update(self, x, y):
        """
        Update the weights of the perceptron based on a single example.
            x is a numpy array with D elements, where D is the value of the
                `dimensions`  parameter passed to Perceptron.__init__
            y is either 1 or -1

        Returns:
            True if the perceptron weights have changed, False otherwise
        """
        
        if self.predict(x) == y: #if prediction is the same as training data then return false (no update)
            return False
        else:
            self.weights += np.dot(y, x) #perceptron update w_new = w + y*x

            return True


    def train(self):
        """
        Train the perceptron until convergence.

        To iterate through all of the data points once (a single epoch), you can
        do:
            for x, y in self.get_data_and_monitor(self):
                ...

        get_data_and_monitor yields data points one at a time. It also takes the
        perceptron as an argument so that it can monitor performance and display
        graphics in between yielding data points.
        """
        
        convergence_count = 1
        while convergence_count != 0:
            convergence_count = 0
            for x, y in self.get_data_and_monitor(self): #convergence_count only returns 0 if there are no updates
                if self.update(x, y):
                    convergence_count += 1
