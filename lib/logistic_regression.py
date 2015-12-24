import numpy as np

class LogisticRegression:
    def predict(self, X):
        raise "Implement in subclass"

    def fit(self, X, y):
        raise "Implement in subclass"


class Binomial(LogisticRegression):
    def __init__(self, options={'iterations': 100, 'alpha': 0.01}):
        self.alpha = options['alpha']
        self.iterations = options['iterations']
        self.theta = None

    def predict(self, X):
        n, dim = X.shape
        x = np.ones([n, dim+1])
        x[: , 1: ] = X

        threshold = self._logistic(x.dot(self.theta))
        return (threshold <= 0.5).astype(int)

    def fit(self, X, y):
        n, dim = X.shape

        self.theta = np.zeros(dim + 1)
        x = np.ones([n, dim+1])
        x[:, 1: ] = X

        for _ in range(self.iterations):
            self.theta -= self.alpha * (y - self._logistic(x.dot(self.theta))).dot(x)

        return self

    def _logistic(self, x):
        return 1 / (1 + np.exp(-x))
