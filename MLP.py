import numpy as np
import pickle
from Layer import Layer  # Import klasy Layer (wcześniej zdefiniowanej warstwy)

class MLP:
    def __init__(self, layer_sizes, use_bias=True):
        self.layers = []  # Lista przechowująca warstwy sieci
        self.layer_sizes = layer_sizes  # Lista rozmiarów warstw, np. [3, 5, 2]
        self.use_bias_globally = use_bias  # Flaga określająca, czy biasy są używane globalnie

        # Tworzenie warstw na podstawie rozmiarów
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], use_bias=use_bias))

    def forward(self, input_data):
        # Propagacja danych wejściowych przez wszystkie warstwy
        current_output = np.array(input_data, dtype=float)
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def backward(self, expected_output, learning_rate, momentum=0.0):
        # Propagacja wsteczna – aktualizacja wag w oparciu o oczekiwane wyjście
        expected_output = np.array(expected_output, dtype=float)
        error_gradient = self.layers[-1].last_output - expected_output  # różnica między predykcją a oczekiwaniem
        for layer in reversed(self.layers):
            error_gradient = layer.backward(error_gradient, learning_rate, momentum)

    def calculate_pattern_error(self, actual_output, expected_output):
        # Obliczenie średniego błędu kwadratowego dla pojedynczego przykładu
        expected_output = np.array(expected_output, dtype=float)
        actual_output = np.array(actual_output, dtype=float)
        return np.mean((expected_output - actual_output) ** 2)

    def get_all_parameters(self):
        # Pobranie wszystkich parametrów sieci (wagi, biasy, wyjścia warstw)
        params = {
            'hidden_outputs': [],
            'hidden_weights': [],
            'hidden_biases': [],
            'output_outputs': None,
            'output_weights': None,
            'output_biases': None
        }

        # Dane warstw ukrytych
        for i, layer in enumerate(self.layers[:-1]):
            if layer.last_output is not None:
                params['hidden_outputs'].append(layer.last_output.copy().tolist())
            params['hidden_weights'].append(layer.weights.copy().tolist())
            if layer.use_bias:
                params['hidden_biases'].append(layer.biases.copy().tolist())
            else:
                params['hidden_biases'].append(np.zeros_like(layer.biases).tolist())

        # Dane warstwy wyjściowej
        output_layer = self.layers[-1]
        if output_layer.last_output is not None:
            params['output_outputs'] = output_layer.last_output.copy().tolist()
        params['output_weights'] = output_layer.weights.copy().tolist()
        if output_layer.use_bias:
            params['output_biases'] = output_layer.biases.copy().tolist()
        else:
            params['output_biases'] = np.zeros_like(output_layer.biases).tolist()

        return params

    def save_network(self, filename):
        # Zapisanie sieci do pliku za pomocą pickle
        network_data = {
            'layer_sizes': self.layer_sizes,
            'use_bias_globally': self.use_bias_globally,
            'layers_data': []
        }

        for layer in self.layers:
            layer_data = {
                'weights': layer.weights,
                'biases': layer.biases,
                'use_bias': layer.use_bias
            }
            network_data['layers_data'].append(layer_data)

        with open(filename, 'wb') as f:
            pickle.dump(network_data, f)
        print(f"Sieć zapisana do pliku: {filename}")

    @classmethod
    def load_network(cls, filename):
        # Wczytanie sieci z pliku pickle
        with open(filename, 'rb') as f:
            network_data = pickle.load(f)

        use_bias_globally = network_data.get('use_bias_globally', True)
        mlp = cls(network_data['layer_sizes'], use_bias=use_bias_globally)

        # Przywrócenie wag i biasów do warstw
        for i, layer_data in enumerate(network_data['layers_data']):
            mlp.layers[i].weights = layer_data['weights']
            mlp.layers[i].biases = layer_data['biases']
            mlp.layers[i].use_bias = layer_data['use_bias']
            # Inicjalizacja prędkości dla momentu
            mlp.layers[i].weight_velocity = np.zeros_like(layer_data['weights'])
            if mlp.layers[i].use_bias:
                mlp.layers[i].bias_velocity = np.zeros_like(layer_data['biases'])
            else:
                mlp.layers[i].bias_velocity = np.zeros_like(layer_data['biases'])

        print(f"Sieć wczytana z pliku: {filename}")
        return mlp
