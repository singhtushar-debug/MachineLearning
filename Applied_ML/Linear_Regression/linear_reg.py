class LR_Model:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.interations = iterations
        self.w = 0
        self.b = 0

    def cost_function(self, x, y):
        """
        Compute the cost(error).
        """
        m = x.shape[0]
        cost = 0
        for i in range(m):
            fw_b = x[i] * self.w + self.b
            cost += (fw_b - y[i]) ** 2
        total = cost / (2 * m)
        return total

    def compute_gradient(self, x, y):
        m = x.shape[0]
        dj_db = 0
        dj_dw = 0
        for i in range(m):
            fw_b = x[i] * self.w + self.b
            dj_dw += (fw_b - y[i]) * x[i]
            dj_db += fw_b - y[i]
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        return dj_dw, dj_db

    def train_model(self, x, y):
        """
        Find the optimal value for w and b using gradient descent
        """
        print(f"Initial cost :{self.cost_function(x, y):.4f}")
        w = self.w
        b = self.b
        for i in range(self.interations):
            dj_dw, dj_db = self.compute_gradient(x, y, w, b)
        self.w = self.w - self.learning_rate * dj_dw
        self.b = self.b - self.learning_rate * dj_db

        print(f"Final cost :{self.cost_function(x, y):.4f}")

    def predict(self, x):
        return self.w * x + self.b
