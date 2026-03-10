import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Helper function for calculating mean squared error
def mean_squared_error(y_true, y_predicted):
        # Calculates the loss/cost
        cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
        return cost
    
# Gradient descent function
def gradient_descent(X, y, iterations, learning_rate, stopping_threshold=1e-6):
    # Initializes weight, bias, learning rate, and the iterations
    current_weight = np.zeros(X.shape[1])  # Handles multiple features, for 2D input
    current_bias = 0.01
    n = float(len(X))
    costs = []
    weights = []
    previous_cost = None
    
    # Gradient Descent for 2D data
    for i in range(iterations):
        # Makes the predictions
        y_predicted = np.dot(X, current_weight) + current_bias
        # Calculates the current cost
        current_cost = mean_squared_error(y, y_predicted)
        # Stops if change in cost is <= the threshold
        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break
        previous_cost = current_cost
        costs.append(current_cost)
        weights.append(current_weight.copy())
        
        # Calculates the gradients
        weight_derivative = -(2/n) * np.dot(X.T, (y - y_predicted))
        bias_derivative = -(2/n) * np.sum(y - y_predicted)
        # Updates weights and bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)
        
        # Prints the parameters for every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i+1}: Cost {current_cost}, Weights {current_weight}, Bias {current_bias}")
            
    return current_weight, current_bias


class LinearRegression():
    
    def __init__(self, learning_rate=0.0001, num_iterations=1000):
        # Hyperparameters for gradient descent
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.coefficients = None
        self.intercept = None
        self.fitted = False
    
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier using Gradient Descent
        
        Args:
            X (array<m,n> or pd.Series): a matrix of floats with
                m rows (#samples) and n columns (#features), or a pandas Series.
            y (array<m> or pd.Series): a vector of floats or a pandas Series
        """
        # Converts eventual series to numpy arrays
        if isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Ensures X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Uses the gradient_descent function to estimate weights and bias
        self.coefficients, self.intercept = gradient_descent(X, y, iterations=self.num_iterations, learning_rate=self.learning_rate)
        self.fitted = True
        
    def predict(self, X):
        """
        Generates predictions
        
        Args:
            X (array<m,n> or pd.Series): a matrix of floats with 
                m rows (#samples) and n columns (#features), or a pandas Series.
            
        Returns:
            A length m array of floats
        """
        # Converts X, if it is a series, to a numpy array
        if isinstance(X, pd.Series):
            X = X.values

        # Ensures X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Ensures the model has been fitted
        if not self.fitted:
            raise ValueError("The model has not been fitted yet.")

        # Generates predictions
        predictions = np.dot(X, self.coefficients) + self.intercept
        
        return predictions

