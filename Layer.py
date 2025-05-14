# Layer.py
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
        # output_error_gradient to dE/da_curr_layer dla warstw ukrytych, lub (y_true - y_pred) dla warstwy wyjściowej
        # delta = (dE/da_curr_layer) * sigmoid_derivative(z_curr_layer)  lub (y_true - y_pred) * sigmoid_derivative(z_output_layer)
        # delta to dE/dz_curr_layer

        sigmoid_deriv = self.sigmoid_derivative(self.last_z)
        delta = output_error_gradient * sigmoid_deriv

        # Gradient dla wag: dE/dW = dE/dz * dz/dW = delta * a_prev_layer
        # self.last_input to a_prev_layer
        # Dla operacji wsadowych (batch) last_input.T @ delta, ale tu jest online
        # Wymiar last_input: (input_size,)
        # Wymiar delta: (output_size,)
        # dW_ij = delta_j * last_input_i
        weight_gradient = np.outer(self.last_input, delta)

        # Aktualizacja wag z momentum
        self.weight_velocity = (momentum * self.weight_velocity) - (learning_rate * weight_gradient)
        self.weights += self.weight_velocity

        # Aktualizacja biasów (jeśli są używane)
        # Gradient dla biasów: dE/db = dE/dz * dz/db = delta * 1 = delta
        if self.use_bias:
            bias_gradient = delta
            self.bias_velocity = (momentum * self.bias_velocity) - (learning_rate * bias_gradient)
            self.biases += self.bias_velocity

        # Propagacja błędu do poprzedniej warstwy: dE/da_prev_layer
        # dE/da_prev_layer = dE/dz_curr_layer @ W_curr_layer.T = delta @ W_curr_layer.T
        error_to_propagate = np.dot(delta, self.weights.T)
        return error_to_propagate

    @staticmethod
    def sigmoid(x):
        # Clipping, aby uniknąć overflow w exp() dla dużych wartości ujemnych x
        # i underflow (exp(-x) -> inf) dla dużych wartości dodatnich x
        clipped_x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-clipped_x))

    @staticmethod
    def sigmoid_derivative(z):
        # Pochodna funkcji sigmoidalnej s'(z) = s(z) * (1 - s(z))
        # Gdzie s(z) to wynik funkcji sigmoid(z)
        s = Layer.sigmoid(z)  # Używamy tej samej funkcji sigmoid co w forward
        return s * (1 - s)