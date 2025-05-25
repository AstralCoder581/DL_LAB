import numpy as np

class MovieDecisionPerceptron:
    def __init__(self):
        # Given weights and bias
        self.weights = np.array([0.2, 0.4, 0.2])  # hero, heroine, climate
        self.bias = -0.5
    
    def predict(self, inputs):
        """Predict decision based on inputs"""
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        return 1 if weighted_sum > 0 else 0
    
    def calculate_accuracy(self, test_cases):
        """Calculate accuracy on test cases"""
        correct = 0
        total = len(test_cases)
        
        for inputs, expected in test_cases:
            prediction = self.predict(inputs)
            if prediction == expected:
                correct += 1
            print(f"Inputs: {inputs}, Predicted: {prediction}, Expected: {expected}")
        
        accuracy = correct / total * 100
        return accuracy

# Example usage
perceptron = MovieDecisionPerceptron()

# Test cases: [hero, heroine, climate], expected_output
test_cases = [
    ([1, 1, 1], 1),  # All favorable
    ([1, 1, 0], 1),  # Hero and heroine good, bad climate
    ([1, 0, 1], 0),  # Hero good, heroine bad, good climate
    ([0, 1, 1], 0),  # Hero bad, heroine good, good climate
    ([0, 0, 0], 0),  # All unfavorable
    ([1, 0, 0], 0),  # Only hero good
    ([0, 1, 0], 0),  # Only heroine good
    ([0, 0, 1], 0),  # Only climate good
]

accuracy = perceptron.calculate_accuracy(test_cases)
print(f"\nAccuracy: {accuracy:.2f}%")

# Single prediction example
inputs = [1, 1, 1]  # Favorite hero, heroine, good climate
decision = perceptron.predict(inputs)
print(f"\nFor inputs {inputs}: {'Go for movie' if decision else 'Stay home'}")
