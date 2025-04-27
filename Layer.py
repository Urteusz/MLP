import numpy as np

class Layer:
    def __init__(self, input_size, output_size, use_bias=True):
        self.weights = np.random.uniform(-0.5, 0.5, (input_size, output_size))
        self.use_bias = use_bias
        if self.use_bias:
            self.biases = np.random.uniform(-0.5, 0.5, (output_size,))
        else:
            self.biases = np.zeros(output_size)
        self.last_input = None
        self.last_output = None

    def forward(self, input_data):
        self.last_input = input_data
        z = np.dot(input_data, self.weights)
        if self.use_bias:
            z += self.biases
        self.last_output = self.sigmoid(z)
        return self.last_output

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
