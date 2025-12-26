import numpy as np

class LinearRegression_():
    def __init__(self, random_state=42):
        self._coef = None
        self._intercept = None

        self.random_state = random_state
        self._rng = np.random.default_rng(self.random_state)

    def _hypothesis(self, X,theta):
        return np.dot(X, theta)

    def _cost(self, X, y, theta):
        m = y.shape[0]
        predictions = self._hypothesis(X, theta)
        sq_errors = (predictions - y)**2 
        J = 1/2 * (1/m)  * np.sum(sq_errors)

        return J
    
    def _gradient_descent(self,X,y,theta, alpha):
        m = y.shape[0]
        predictions = self._hypothesis(X,theta)
        errors = predictions - y

        theta = theta - (alpha/m) * (X.T.dot(errors))

        J = 1/2 * (1/m)  * np.sum(errors**2)

        return theta, J
    
    def fit(self, X, y, num_iters=600, alpha=0.001):
        self.cost_history = []
        m,n = X.shape
        theta = self._rng.random(n+1)

        X = np.c_[np.ones(m), X]
        y = y.ravel()

        for _ in range(num_iters):
            theta, cost = self._gradient_descent(X, y ,theta, alpha)
            self.cost_history.append(cost)

        self._intercept = theta[0]
        self._coef = theta[1:]
        
        return self
    
    def predict(self, X):
        return X.dot(self._coef) + self._intercept