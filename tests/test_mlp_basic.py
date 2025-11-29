import numpy as np
from MLP import MLP

def test_forward_shape():
    mlp = MLP([4, 5, 3], use_bias=True)
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = mlp.forward(x)
    assert y.shape == (3,)
    assert np.all((y >= 0) & (y <= 1))

def test_backward_reduces_error():
    mlp = MLP([2, 3, 1], use_bias=True)
    # Prosty zbiór: funkcja XOR podana jako regresja (0/1)
    data = [
        (np.array([0.0, 0.0]), np.array([0.0])),
        (np.array([0.0, 1.0]), np.array([1.0])),
        (np.array([1.0, 0.0]), np.array([1.0])),
        (np.array([1.0, 1.0]), np.array([0.0]))
    ]
    # Kilka kroków uczenia i sprawdzenie czy błąd średni maleje
    initial_error = 0.0
    for x, y_exp in data:
        y_act = mlp.forward(x)
        initial_error += mlp.calculate_pattern_error(y_act, y_exp)
    initial_error /= len(data)

    for epoch in range(50):
        for x, y_exp in data:
            y_act = mlp.forward(x)
            mlp.backward(y_exp, learning_rate=0.5, momentum=0.1)

    final_error = 0.0
    for x, y_exp in data:
        y_act = mlp.forward(x)
        final_error += mlp.calculate_pattern_error(y_act, y_exp)
    final_error /= len(data)

    assert final_error < initial_error, f"Error did not decrease: {initial_error} -> {final_error}"
