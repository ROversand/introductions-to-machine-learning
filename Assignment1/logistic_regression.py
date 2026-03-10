import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss function
def binary_cross_entropy(y_true, y_predicted):
    # Avoids log(0) and NaN values by clipping
    y_predicted = np.clip(y_predicted, 1e-10, 1 - 1e-10)
    log_loss = -np.mean(y_true * np.log(y_predicted) + (1 - y_true) * np.log(1 - y_predicted))
    return log_loss

# Gradient descent for the logistic regression
def gradient_descent(X, y, iterations, learning_rate, stopping_threshold=1e-12):
    # Initializes weight, bias, learning rate, and the iterations
    current_weight = np.zeros(X.shape[1])
    current_bias = 0.1
    n = float(len(X))
    costs = []
    previous_cost = None
    
    for i in range(iterations):
        # Linear combination
        linear_model = np.dot(X, current_weight) + current_bias
        # Applies sigmoid function to get predicted probabilites
        y_predicted = sigmoid(linear_model)
        
        # Calculates the cost with the binary cross-entropy loss function
        current_cost = binary_cross_entropy(y, y_predicted)
        
        # Break if the change in cost is <= the threshold
        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break
        previous_cost = current_cost
        costs.append(current_cost)
        
        # Calculates the gradients
        weight_derivative = -(2/n) * np.dot(X.T, (y - y_predicted))
        bias_derivative = -(2/n) * np.sum(y - y_predicted)
        
        # Updates the weights and bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)
        
        # Prints parameters for every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {current_cost}, Weights = {current_weight}, Bias = {current_bias}")
            
    return current_weight, current_bias

class LogisticRegression:
    
    def __init__(self, learning_rate=0.1, num_iterations=1000):
        #Hyperparameters for the gradient descent
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.coefficients = None
        self.intercept = None
        self.fitted = False
        
    def fit(self, X, y):
        """
        Fits the model to estimate parameters using gradient descent.
        
        Args:
            X (array<m,n> or pd.Series): Input matrix with m samples and n features.
            y (array<m> or pd.Series): Output binary labels (0 or 1).
        """
        # Converts eventual series to numpy arrays
        if isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Estimates weights and bias with the gradient descent function
        self.coefficients, self.intercept = gradient_descent(X, y, iterations=self.num_iterations, learning_rate=self.learning_rate)
        self.fitted = True
    
    def predict_probabilites(self, X):
        """
        Predicts the probabilities for the input features.
        
        Args:
            X (array<m,n> or pd.Series): Input matrix with m samples and n features.
            
        Returns:
            A length m array of probabilities.
        """
        # Converts X, if it is a series, to a numpy array
        if isinstance(X, pd.Series):
            X = X.values
            
        # Ensures X is 2D
        if (len(X.shape) == 1):
            X = X.reshape(-1, 1)
        
        # Ensures the model has been fitted
        if not self.fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Calculates linear model and applies sigmoid to get the probabilites
        linear_model = np.dot(X, self.coefficients) + self.intercept
        probabilites = sigmoid(linear_model)
        return probabilites
    
    def predict(self, X):
        """
        Predicts the binary class (0 or 1) for the input features.
        
        Args:
            X (array<m,n> or pd.Series): Input matrix with m samples and n features.
            
        Returns:
            A length m array of binary predictions (0 or 1).
        """
        probabilities = self.predict_probabilites(X)
        
        # Converts probabilities to binary predictions
        predictions = [1.0 if probability >= 0.5 else 0.0 for probability in probabilities]
        
        return np.array(predictions)
    
        
            
        