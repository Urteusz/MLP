import numpy as np


class Layer:
    def __init__(self, input_size, output_size, use_bias=True):
        # Inicjalizacja wag losowymi wartościami z zakresu [-0.5, 0.5]
        self.weights = np.random.uniform(-0.5, 0.5, (input_size, output_size))

        self.use_bias = use_bias
        if self.use_bias:
            # Inicjalizacja biasów losowymi wartościami (jeden bias na neuron wyjściowy)
            self.biases = np.random.uniform(-0.5, 0.5, (output_size,))
        else:
            # Gdy biasy są wyłączone, przypisujemy wektor zer
            self.biases = np.zeros(output_size)

        # Zapamiętanie wartości potrzebnych do propagacji wstecznej
        self.last_input = None  # dane wejściowe z ostatniego kroku forward
        self.last_output = None  # wynik po aktywacji
        self.last_z = None  # wynik przed aktywacją (czyli input * wagi + biasy)

        # Inicjalizacja wektorów prędkości do optymalizacji z użyciem momentum
        self.weight_velocity = np.zeros_like(self.weights)
        if self.use_bias:
            self.bias_velocity = np.zeros_like(self.biases)
        else:
            self.bias_velocity = np.zeros_like(self.biases)

    def forward(self, input_data):
        # Propagacja w przód: zapisanie danych wejściowych
        self.last_input = input_data

        # Obliczenie iloczynu wejścia i wag
        self.last_z = np.dot(input_data, self.weights)

        # Dodanie biasów, jeśli są używane
        if self.use_bias:
            self.last_z += self.biases

        # Zastosowanie funkcji aktywacji sigmoid
        self.last_output = self.sigmoid(self.last_z)

        return self.last_output

    def backward(self, output_error_gradient, learning_rate, momentum=0.0):
        # Obliczenie pochodnej funkcji sigmoid
        sigmoid_deriv = self.last_output * (1 - self.last_output)

        # delta to lokalny gradient błędu
        delta = output_error_gradient * sigmoid_deriv

        # Obliczenie gradientu wag (macierz zewnętrznego iloczynu wejścia i delty)
        weight_gradient = np.outer(self.last_input, delta)

        # Aktualizacja wag z wykorzystaniem momentu
        self.weight_velocity = (momentum * self.weight_velocity) - (learning_rate * weight_gradient)
        self.weights += self.weight_velocity

        # Aktualizacja biasów (jeśli są używane), również z momentem
        if self.use_bias:
            bias_gradient = delta  # biasy mają gradient równy delcie
            self.bias_velocity = (momentum * self.bias_velocity) - (learning_rate * bias_gradient)
            self.biases += self.bias_velocity

        # Obliczenie błędu do propagacji do poprzedniej warstwy
        error_to_propagate = np.dot(delta, self.weights.T)

        return error_to_propagate

    @staticmethod
    def sigmoid(x):
        # Zabezpieczenie przed przepełnieniem w funkcji exp
        clipped_x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-clipped_x))
