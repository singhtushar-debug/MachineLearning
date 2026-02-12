import numpy as np


class Multiple_LR_Model:
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize the Linear Regression model.

        Args:
            learning_rate : Learning rate.Defaults to 0.01.
            iterations : Number of iteratoins for training.Defaults to 1000.
        """
        self.learning_rate = learning_rate
        self.interations = iterations
        self.w = None
        self.b = None

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
            fw_b = np.dot(x[i], w) + b
            cost += (fw_b - y[i]) ** 2
        total = cost / (2 * m)
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
        dj_db = 0
        dj_dw = np.zeros((n,))
        for i in range(m):
            fw_b = np.dot(x[i], w) + b
            for j in range(n):
                dj_dw[j] += (fw_b - y[i]) * x[i, j]
            dj_db += fw_b - y[i]
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        return dj_dw, dj_db

    def train_model(self, x, y):
        """
        Trains the model to the training data using gradient descent.

        Args:
            x (ndarray): Training data. Shape (m,n).
            y (ndarray): Traget values. Shape (m,).

        Returns:
            w (ndarray): Final wegihts calculated by gradient descent.
            b (float): Final bias calculated by gradient descent.
        """
        _, n = x.shape
        self.w = np.zeros(n)
        self.b = 0
        print(f"Initial cost :{self.cost_function(x, y, self.w, self.b):.4f}")
        for i in range(self.interations):
            dj_dw, dj_db = self.compute_gradient(x, y, self.w, self.b)
            self.w = self.w - self.learning_rate * dj_dw
            self.b = self.b - self.learning_rate * dj_db

        print(f"Final cost :{self.cost_function(x, y, self.w, self.b):.4f}")
        return self.w, self.b

    def predict(self, x):
        """
        Predicts target values for new input data.

        Args:
            x (ndarray): Input data. shape (m,n) or (n,).

        Returns:
            ndarray : Predicted value(s) using the linear regression model (np.dot(w,x) + b).

        """
        return np.dot(x, self.w) + self.b
