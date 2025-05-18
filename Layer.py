import numpy as np


class Layer:
    def __init__(self, input_size, output_size, use_bias=True):
        # Inicjalizacja wag z przedziału [-0.5, 0.5]
        self.weights = np.random.uniform(-0.5, 0.5, (input_size, output_size))
        self.use_bias = use_bias
        if self.use_bias:
            self.biases = np.random.uniform(-0.5, 0.5, (output_size,))
        else:
            # Nawet jeśli bias nie jest używany, inicjalizujemy go zerami dla spójności kształtów
            self.biases = np.zeros(output_size)

        self.last_input = None
        self.last_output = None
        self.last_z = None  # Suma ważona przed aktywacją

        # Do mechanizmu momentum
        self.weight_velocity = np.zeros_like(self.weights)
        if self.use_bias:
            self.bias_velocity = np.zeros_like(self.biases)
        else:
            # Jeśli bias nie jest używany, velocity dla biasu też jest zerowe
            self.bias_velocity = np.zeros_like(self.biases)

    def forward(self, input_data):
        self.last_input = input_data
        self.last_z = np.dot(input_data, self.weights)
        if self.use_bias:
            self.last_z += self.biases
        self.last_output = self.sigmoid(self.last_z)
        return self.last_output

    def backward(self, output_error_gradient, learning_rate, momentum=0.0):
        sigmoid_deriv = self.last_output * (1 - self.last_output)
        delta = output_error_gradient * sigmoid_deriv

        weight_gradient = np.outer(self.last_input, delta)

        # Aktualizacja wag z momentum
        self.weight_velocity = (momentum * self.weight_velocity) - (learning_rate * weight_gradient)
        self.weights += self.weight_velocity

        if self.use_bias:
            bias_gradient = delta
            self.bias_velocity = (momentum * self.bias_velocity) - (learning_rate * bias_gradient)
            self.biases += self.bias_velocity

        error_to_propagate = np.dot(delta, self.weights.T)
        return error_to_propagate

    @staticmethod
    def sigmoid(x):
        clipped_x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-clipped_x))