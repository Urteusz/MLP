import numpy as np
from MLP import MLP
from Trainer import Trainer


def run_autoencoder_experiments(
        bias_value=0.6,
        momentum_value=0.0,
        epochs=None,
        error_threshold=None,
        LOG_INTERVAL_AUTOENCODER=100,
        log_filename="autoencoder_train_log.csv",
        use_bias=True
):
    AUTOENCODER_ARCHITECTURE = [4, 2, 4]
    AUTOENCODER_PATTERNS = [
        (np.array([1, 0, 0, 0], dtype=float), np.array([1, 0, 0, 0], dtype=float)),
        (np.array([0, 1, 0, 0], dtype=float), np.array([0, 1, 0, 0], dtype=float)),
        (np.array([0, 0, 1, 0], dtype=float), np.array([0, 0, 1, 0], dtype=float)),
        (np.array([0, 0, 0, 1], dtype=float), np.array([0, 0, 0, 1], dtype=float))
    ]

    print("--- Eksperyment z Autoenkoderem ---")
    print(f"Bias: {'TAK' if use_bias else 'NIE'} | LR: {bias_value}, Momentum: {momentum_value}")

    mlp_auto = MLP(layer_sizes=AUTOENCODER_ARCHITECTURE, use_bias=use_bias)
    trainer_auto = Trainer(mlp_auto, AUTOENCODER_PATTERNS,
                           learning_rate=bias_value,
                           momentum=momentum_value,
                           shuffle=True)

    if epochs is None:
        trainer_auto.train(error_threshold=error_threshold,
                           log_interval=LOG_INTERVAL_AUTOENCODER,
                           log_file=log_filename)
    else:
        trainer_auto.train(max_epochs=epochs,
                           log_interval=LOG_INTERVAL_AUTOENCODER,
                           log_file=log_filename)

    # Ewaluacja końcowego błędu
    final_epoch_error = 0
    for x, y_exp in AUTOENCODER_PATTERNS:
        y_act = mlp_auto.forward(x)
        final_epoch_error += mlp_auto.calculate_pattern_error(y_act, y_exp)
    avg_final_error = final_epoch_error / len(AUTOENCODER_PATTERNS)
    print(f"Finalny średni błąd MSE: {avg_final_error:.8f}")

    print("\nWartości na wyjściach warstwy ukrytej i końcowej:")
    if mlp_auto.layers:
        hidden_layer_index = 0

        # Zbieranie wyników dla wszystkich wzorców
        hidden_outputs_matrix = []
        final_outputs_matrix = []

        for i, (pattern_in, _) in enumerate(AUTOENCODER_PATTERNS):
            output = mlp_auto.forward(pattern_in)

            # Wartości warstwy ukrytej (pierwsza warstwa)
            hidden_outputs = mlp_auto.layers[hidden_layer_index].last_output
            hidden_outputs_matrix.append(hidden_outputs)

            # Wartości warstwy końcowej (output sieci)
            final_outputs_matrix.append(output)

        # Konwersja do macierzy numpy
        hidden_matrix = np.array(hidden_outputs_matrix)
        final_matrix = np.array(final_outputs_matrix)

        print("\nWarstwa ukryta (4x2):")
        print(np.round(hidden_matrix, 4))

        print("\nWarstwa końcowa (4x4):")
        print(np.round(final_matrix, 4))

    # Zapis modelu
    suffix = "on" if use_bias else "off"
    mlp_auto.save_network(f"autoencoder_model_bias_{suffix}.pkl")


if __name__ == "__main__":
    print("Współczynnik nauki - 0,9; współczynnik momentum - 0,0:")
    run_autoencoder_experiments(bias_value=0.9, momentum_value=0.0, error_threshold=0.01)

    print("\nWspółczynnik nauki - 0,6; współczynnik momentum - 0,0:")
    run_autoencoder_experiments(bias_value=0.6, momentum_value=0.0, error_threshold=0.01)

    print("\nWspółczynnik nauki - 0,2; współczynnik momentum - 0,0:")
    run_autoencoder_experiments(bias_value=0.2, momentum_value=0.0, error_threshold=0.01)

    print("\nWspółczynnik nauki - 0,9; współczynnik momentum - 0,6:")
    run_autoencoder_experiments(bias_value=0.9, momentum_value=0.6, error_threshold=0.01)

    print("\nWspółczynnik nauki - 0,2; współczynnik momentum - 0,9:")
    run_autoencoder_experiments(bias_value=0.2, momentum_value=0.9, error_threshold=0.01)