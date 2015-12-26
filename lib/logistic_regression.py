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
        x = self._add_ones_to(X)

        threshold = self._logistic(x.dot(self.theta))
        return np.rint(threshold)

    def fit(self, X, y):
        n, dim = X.shape

        self.theta = np.zeros(dim + 1)
        x = self._add_ones_to(X)

        for _ in range(self.iterations):
            self.theta += self.alpha * x.T.dot(y - self._logistic(x.dot(self.theta)))

        return self

    def _logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def _add_ones_to(self, X):
        n, dim = X.shape
        x = np.ones([n, dim+1])
        x[: , 1: ] = X

        return x


class Multinomial(LogisticRegression):
    def __init__(self, options={'iterations': 1000, 'alpha': 0.01}):
        self.regressors = []
        self.options = options

    def predict(self, X):
        x = self._add_ones_to(X)

        probabilities = self._compute_probabilities(self.regressors, x)

        return np.argmax(probabilities, axis=0)

    def fit(self, X, y):
        self.klasses = np.unique(y)

        for k in self.klasses[0:-1]:
            y_ = (y == k).astype(int)
            self.regressors.append(self._get_regressor_for(X, y_))

        return self

    def _get_regressor_for(self, X, y):
        regressor = Binomial(self.options)
        regressor.fit(X, y)

        return regressor

    def _add_ones_to(self, X):
        n, dim = X.shape
        x = np.ones([n, dim+1])
        x[: , 1: ] = X

        return x

    def _compute_probabilities(self, regressors, x):
        n, dim = x.shape
        numerators = np.array([]).reshape(0, n)
        probabilities = np.array([]).reshape(0, n)

        for regressor in self.regressors:
            numerators = np.vstack([numerators, np.exp(x.dot(regressor.theta))])

        for numerator in numerators:
            probabilities = np.vstack([probabilities, numerator / (1 + np.sum(numerators, axis=0))])
        probabilities = np.vstack([probabilities, 1 / (1 + np.sum(numerators, axis=0))])

        return probabilities

    def _logistic(self, x):
        return 1 / (1 + np.exp(-x))
