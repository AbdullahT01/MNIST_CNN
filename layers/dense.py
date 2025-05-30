import numpy as np

class Dense:
    def __init__(self, input_len, output_len):
        self.weights = np.random.randn(input_len, output_len) * 0.1
        self.biases = np.zeros(output_len)

    def forward(self, input):
        self.last_input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, d_out, learning_rate):
        # Reshape d_out to (output_len, 1)
        d_out = d_out.reshape(-1, 1)  # (10, 1)

        # Reshape last input to (1, input_len)
        last_input_reshaped = self.last_input.reshape(1, -1)  # (1, 1352)

        # Compute gradients
        d_weights = np.dot(last_input_reshaped.T, d_out.T)  # (1352, 10)
        d_biases = d_out.flatten()  # (10,)

        # Update weights and biases
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        # Compute gradient w.r.t. input to this layer
        d_input = np.dot(self.weights, d_out).flatten()  # (1352,)

        return d_input
