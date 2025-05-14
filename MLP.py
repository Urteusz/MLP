from Layer import Layer

class MLP:
    def __init__(self, layer_sizes, use_bias=True):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], use_bias))

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    def backward(self, expected_output, learning_rate, momentum=0.0):
        output_error = expected_output - self.layers[-1].last_output
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate, momentum)

    def train(self, x, y, learning_rate=0.6, momentum=0.0):
        # Forward pass
        output = self.forward(x)

        # Compute error at output layer
        error = y - output
        total_error = np.mean((error) ** 2)

        # Backward pass
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate, momentum)

        return total_error