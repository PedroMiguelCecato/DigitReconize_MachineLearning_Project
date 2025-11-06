import numpy as np

class RegressaoLogistica:
    def __init__(self, eta=0.01, tmax=5000, lambda_=0.01):
        self.eta = eta
        self.tmax = tmax
        self.lambda_ = lambda_
        self.w = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def execute(self, X, y):
        X = np.array(X)
        y = np.where(np.array(y) <= 0, -1, 1)

        X = np.c_[np.ones((X.shape[0], 1)), X]
        n_amostras, n_features = X.shape
        self.w = np.zeros(n_features)

        for _ in range(self.tmax):
            z = X @ self.w
            grad = -(1/n_amostras) * (X.T @ (y / (1 + np.exp(y * z))))
            grad[1:] += self.lambda_ * self.w[1:]
            self.w -= self.eta * grad

    def predict_prob(self, X):
        X = np.array(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X @ self.w)

    def predict(self, X):
        X = np.array(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return np.sign(X @ self.w)

    def get_w(self):
        return self.w