# MLP.py
import numpy as np
import pickle
from Layer import Layer  # Upewnij się, że Layer.py jest w tym samym katalogu


class MLP:
    def __init__(self, layer_sizes, use_bias=True):
        self.layers = []
        self.layer_sizes = layer_sizes
        self.use_bias_globally = use_bias  # To jest globalne ustawienie dla nowych warstw

        for i in range(len(layer_sizes) - 1):
            # Każda warstwa może indywidualnie decydować o biasie, ale tu ustawiamy globalnie
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], use_bias=use_bias))

    def forward(self, input_data):
        current_output = np.array(input_data, dtype=float)
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def backward(self, expected_output, learning_rate, momentum=0.0):
        # Konwersja na numpy array dla spójności
        expected_output = np.array(expected_output, dtype=float)

        error_gradient = self.layers[-1].last_output - expected_output

        # Propagujemy błąd wstecz przez każdą warstwę, zaczynając od ostatniej
        for layer in reversed(self.layers):
            error_gradient = layer.backward(error_gradient, learning_rate, momentum)

    def calculate_pattern_error(self, actual_output, expected_output):
        """Oblicza błąd MSE dla pojedynczego wzorca."""
        expected_output = np.array(expected_output, dtype=float)
        actual_output = np.array(actual_output, dtype=float)
        return np.mean((expected_output - actual_output) ** 2)

    def get_all_parameters(self):
        params = {
            'hidden_outputs': [],
            'hidden_weights': [],
            'hidden_biases': [],
            'output_outputs': None,
            'output_weights': None,
            'output_biases': None
        }

        # Warstwy ukryte
        for i, layer in enumerate(self.layers[:-1]):
            if layer.last_output is not None:
                params['hidden_outputs'].append(layer.last_output.copy().tolist())
            params['hidden_weights'].append(layer.weights.copy().tolist())
            if layer.use_bias:
                params['hidden_biases'].append(layer.biases.copy().tolist())
            else:
                # Zapisujemy pustą listę lub listę zer, jeśli bias nie jest używany
                params['hidden_biases'].append(np.zeros_like(layer.biases).tolist())

        # Warstwa wyjściowa
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
        network_data = {
            'layer_sizes': self.layer_sizes,
            'use_bias_globally': self.use_bias_globally,  # Zapisujemy globalne ustawienie
            'layers_data': []
        }
        for layer in self.layers:
            layer_data = {
                'weights': layer.weights,
                'biases': layer.biases,
                'use_bias': layer.use_bias  # Indywidualne ustawienie biasu dla warstwy
            }
            network_data['layers_data'].append(layer_data)

        with open(filename, 'wb') as f:
            pickle.dump(network_data, f)
        print(f"Sieć zapisana do pliku: {filename}")

    @classmethod
    def load_network(cls, filename):
        with open(filename, 'rb') as f:
            network_data = pickle.load(f)

        # Używamy use_bias_globally jeśli jest, inaczej domyślne True
        use_bias_globally = network_data.get('use_bias_globally', True)
        mlp = cls(network_data['layer_sizes'], use_bias=use_bias_globally)

        for i, layer_data in enumerate(network_data['layers_data']):
            mlp.layers[i].weights = layer_data['weights']
            mlp.layers[i].biases = layer_data['biases']
            # Ważne: przywracamy indywidualne ustawienie use_bias dla każdej warstwy
            mlp.layers[i].use_bias = layer_data['use_bias']
            # Reinicjalizujemy velocities, bo nie są zapisywane
            mlp.layers[i].weight_velocity = np.zeros_like(layer_data['weights'])
            if mlp.layers[i].use_bias:
                mlp.layers[i].bias_velocity = np.zeros_like(layer_data['biases'])
            else:
                mlp.layers[i].bias_velocity = np.zeros_like(layer_data['biases'])

        print(f"Sieć wczytana z pliku: {filename}")
        return mlp