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

        self.weight_velocity = np.zeros_like(self.weights)
        self.bias_velocity = np.zeros_like(self.biases)

    def forward(self, input_data):
        self.last_input = input_data
        z = np.dot(input_data, self.weights)
        if self.use_bias:
            z += self.biases
        self.last_output = self.sigmoid(z)
        return self.last_output

    def backward(self, output_error, learning_rate, momentum=0.0):
        sigmoid_derivative = self.last_output * (1 - self.last_output)
        delta = output_error * sigmoid_derivative

        input_T = np.atleast_2d(self.last_input).T
        delta_T = np.atleast_2d(delta)

        weight_gradient = np.dot(input_T, delta_T)
        self.weight_velocity = momentum * self.weight_velocity - learning_rate * weight_gradient
        self.weights += self.weight_velocity

        if self.use_bias:
            self.bias_velocity = momentum * self.bias_velocity - learning_rate * delta
            self.biases += self.bias_velocity

        return np.dot(delta, self.weights.T)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
