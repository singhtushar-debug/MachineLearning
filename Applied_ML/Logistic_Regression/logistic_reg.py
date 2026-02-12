import numpy as np


class Logistic_reg_mode:
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize the Logistic Regression model.

        Args:
            learning_rate : Learning rate.Defaults to 0.01.
            iterations : Number of iteratoins for training.Defaults to 1000.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None

    def sigmoid(self, z):
        """
        Computes the sigmoid of z.

        Args:
            z (ndarray or float): The input value.

        Returns:
            g (ndarray or float): 1/(1 + exp(-z)).
        """
        g = 1 / (1 + np.exp(-z))
        return g

    def cost_function(self, x, y, w, b):
        """
        Compute the cost(squared error).

        Args:
            x (ndarray): Data, m examples with n features. Shape(m,n).
            y (ndarray): Target values. Shape(m,).
            w (ndarray): Model weights. Shape(n,).
            b (float): Model bias.

        Returns:
            cost (float): cost(sqaured error)
        """
        m = x.shape[0]
        cost = 0
        for i in range(m):
            z_i = np.dot(x[i], w) + b
            f_wb_i = self.sigmoid(z_i)
            cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
        total = cost / m
        return total

    def compute_gradient(self, x, y, w, b):
        """
        Computes the gradient for linear regression.

        Args:
            x (ndarray): Data, m examples with n features. Shape(m,n).
            y (ndarray): Target values. Shape(m,).
            w (ndarray): Model weights. Shape(n,).
            b (float): Model bias.

        Returns:
            dj_dw : The gradient of the cost w.r.t. the weights. Shape(n,).
            dj_db : The gradient of the cost w.r.t. the bias.
        """
        m, n = x.shape
        dj_dw = np.zeros((n,))
        dj_db = 0
        for i in range(m):
            f_wb_i = self.sigmoid(np.dot(x[i], w) + b)
            for j in range(n):
                dj_dw[j] += (f_wb_i - y[i]) * x[i, j]
            dj_db += f_wb_i - y[i]
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db

    def train_model(self, x, y):
        """
        Trains the model to the training data using gradient descent.

        Args:
            x (ndarray): Training data. Shape (m,n).
            y (ndarray): Traget values. Shape (m,).
        """
        _, n = x.shape
        self.w = np.zeros(n)
        self.b = 0
        print(f"Initial cost :{self.cost_function(x, y, self.w, self.b):.4f}")
        for i in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient(x, y, self.w, self.b)
            self.w = self.w - self.learning_rate * dj_dw
            self.b = self.b - self.learning_rate * dj_db

        print(f"Final cost :{self.cost_function(x, y, self.w, self.b):.4f}")
        return self.w, self.b

    def predict(self, x):
        """
        Predicts binary classes for the input data.

        Args:
            x (ndarray): Data for prediction.

        Returns:
            predictions (ndarray): Array of 0s and 1s (predicted class).
        """
        m = x.shape[0]
        predictions = np.zeros(m)
        for i in range(m):
            z_i = np.dot(x[i], self.w) + self.b
            f_wb_i = self.sigmoid(z_i)
            predictions[i] = 1 if f_wb_i >= 0.5 else 0
        return predictions
