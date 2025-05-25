import numpy as np
import matplotlib.pyplot as plt

class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def plot_activation_functions(self):
        x = np.linspace(-10, 10, 1000)
        
        plt.figure(figsize=(15, 10))
        
        # Sigmoid
        plt.subplot(2, 3, 1)
        plt.plot(x, self.sigmoid(x))
        plt.title('Sigmoid')
        plt.grid(True)
        
        # Tanh
        plt.subplot(2, 3, 2)
        plt.plot(x, self.tanh(x))
        plt.title('Tanh')
        plt.grid(True)
        
        # ReLU
        plt.subplot(2, 3, 3)
        plt.plot(x, self.relu(x))
        plt.title('ReLU')
        plt.grid(True)
        
        # Leaky ReLU
        plt.subplot(2, 3, 4)
        plt.plot(x, self.leaky_relu(x))
        plt.title('Leaky ReLU')
        plt.grid(True)
        
        # Softmax (for a single input)
        plt.subplot(2, 3, 5)
        softmax_vals = [self.softmax(np.array([xi, 0, 0]))[0] for xi in x]
        plt.plot(x, softmax_vals)
        plt.title('Softmax (first component)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
af = ActivationFunctions()
af.plot_activation_functions()

# Test individual functions
print(f"Sigmoid(1): {af.sigmoid(1)}")
print(f"ReLU(-2): {af.relu(-2)}")
print(f"Softmax([1,2,3]): {af.softmax(np.array([1,2,3]))}")
