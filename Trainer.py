import numpy as np
import random

class Trainer:
    def __init__(self, mlp, training_data, learning_rate=0.6, momentum=0.0, shuffle=True):
        self.mlp = mlp
        self.training_data = training_data  # list of (input, target) tuples
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.shuffle = shuffle

    def train(self, max_epochs=1000, error_threshold=None, log_interval=10, log_file="training_log.txt"):
        with open(log_file, 'w') as log:
            for epoch in range(max_epochs):
                total_error = 0.0

                if self.shuffle:
                    random.shuffle(self.training_data)

                for x, y in self.training_data:
                    error = self.mlp.train(x, y, self.learning_rate, self.momentum)
                    total_error += error

                avg_error = total_error / len(self.training_data)

                if epoch % log_interval == 0 or epoch == max_epochs - 1:
                    log.write(f"Epoch {epoch}, AvgError: {avg_error:.6f}\n")
                    print(f"Epoch {epoch}, AvgError: {avg_error:.6f}")

                if error_threshold is not None and avg_error <= error_threshold:
                    print(f"Training stopped: error {avg_error:.6f} ≤ threshold {error_threshold}")
                    break
