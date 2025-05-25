import numpy as np

class LearningRules:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate
    
    def hebbian(self, X, y, epochs=10):
        """Vectorized Hebbian learning: Δw = η * sum(x*y) over all samples"""
        return self.lr * epochs * (X * y[:, None]).sum(axis=0), self.lr * epochs * y.sum()
    
    def perceptron(self, X, y, epochs=100):
        """Perceptron learning with vectorized operations"""
        w, b = np.zeros(X.shape[1]), 0
        for _ in range(epochs):
            for x, t in zip(X, y):
                err = t - (1 if x@w + b > 0 else 0)
                w += self.lr * err * x
                b += self.lr * err
        return w, b
    
    def delta(self, X, y, epochs=100):
        """Delta rule using matrix operations"""
        w, b = np.random.randn(X.shape[1])*0.01, 0
        for _ in range(epochs):
            for x, t in zip(X, y):
                err = t - (x@w + b)
                w += self.lr * err * x
                b += self.lr * err
        return w, b
    
    def correlation(self, X, y):
        """Vectorized correlation implementation"""
        return np.array([np.corrcoef(X[:,i], y)[0,1] for i in range(X.shape[1])])
    
    def outstar(self, X, y, epochs=100):
        """Optimized outstar learning"""
        w = np.random.randn(X.shape[1])*0.01
        for _ in range(epochs):
            for x, t in zip(X, y):
                w += self.lr * (x - w) * t
        return w

# Example usage
X = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
y = np.array([1, -1, -1, -1])  # AND gate

rules = LearningRules(0.1)
print("Hebbian:", rules.hebbian(X, y))
print("Perceptron:", rules.perceptron(X, y))
